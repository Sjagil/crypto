"""Fail-closed autonomous trading control plane.

This module joins existing research evidence, point-in-time market data and
live authority.  It deliberately does not submit orders.  Only the canonical
execution client may do that after this control plane has produced an
actionable opportunity and every live preflight gate has passed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from config.settings import Settings
from core.daily_profit_target import confirmed_capital_flow_for_date
from core.live_capital import submit_level_2_buy_atomically
from data.market_data import load_ohlcv
from research.portfolio_selection import RotationPortfolioPolicy
from research.residual_reversal import (
    RESIDUAL_REVERSAL_ENGINE_VERSION,
    ResidualReversalParameters,
    backtest_residual_reversal,
    residual_reversal_parameter_set,
)
from utils.common import atomic_write_json, read_json, sha256_file, stable_hash

PRIMARY_STRATEGY_ID = "RR_B60_H5_Z20"
PRIMARY_STRATEGY_DNA = "4571ae8e81aeb4299367643922061e2eabb6523c892ec9a63f08d33f32a939d0"
PRIMARY_TIMEFRAME = "1d"
CONTROL_PLANE_VERSION = "1.0.0"


class MarketRegime(StrEnum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE_LOW_VOL = "RANGE_LOW_VOL"
    RANGE_HIGH_VOL = "RANGE_HIGH_VOL"
    BREAKOUT_EXPANSION = "BREAKOUT_EXPANSION"
    MEAN_REVERSION = "MEAN_REVERSION"
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    LIQUIDITY_STRESS = "LIQUIDITY_STRESS"
    UNCLASSIFIED = "UNCLASSIFIED"


class LiveApprovalStatus(StrEnum):
    APPROVED = "APPROVED"
    NOT_APPROVED = "NOT_APPROVED"
    STRATEGY_MISSING = "STRATEGY_MISSING"
    DNA_MISMATCH = "DNA_MISMATCH"
    INVALID_POLICY = "INVALID_POLICY"


@dataclass(frozen=True, slots=True)
class LiveStrategyApproval:
    strategy_id: str
    strategy_dna_hash: str
    strategy_family: str
    timeframe: str
    approved_markets: tuple[str, ...]
    approved_for_live: bool
    approval_reference: str | None
    approved_at: str | None
    maximum_order_eur: float
    maximum_total_exposure_eur: float
    maximum_open_positions: int
    maximum_new_orders_per_day: int
    autoscale: bool
    spot_only: bool

    @classmethod
    def from_mapping(
        cls,
        strategy_id: str,
        values: Mapping[str, Any],
    ) -> "LiveStrategyApproval":
        return cls(
            strategy_id=strategy_id,
            strategy_dna_hash=str(values["strategy_dna_hash"]),
            strategy_family=str(values["strategy_family"]),
            timeframe=str(values["timeframe"]),
            approved_markets=tuple(
                str(value).upper().replace("/", "-") for value in values["approved_markets"]
            ),
            approved_for_live=bool(values["approved_for_live"]),
            approval_reference=(
                str(values["approval_reference"]) if values.get("approval_reference") else None
            ),
            approved_at=(str(values["approved_at"]) if values.get("approved_at") else None),
            maximum_order_eur=float(values["maximum_order_eur"]),
            maximum_total_exposure_eur=float(values["maximum_total_exposure_eur"]),
            maximum_open_positions=int(values["maximum_open_positions"]),
            maximum_new_orders_per_day=int(values["maximum_new_orders_per_day"]),
            autoscale=bool(values["autoscale"]),
            spot_only=bool(values["spot_only"]),
        )

    def validate(self) -> None:
        if len(self.strategy_dna_hash) != 64:
            raise ValueError("approval strategy DNA must be a SHA-256 hash")
        if not self.approved_markets:
            raise ValueError("approval requires at least one market")
        if not 0 < self.maximum_order_eur <= 10:
            raise ValueError("approval order cap must not exceed EUR 10")
        if not 0 < self.maximum_total_exposure_eur <= 10:
            raise ValueError("approval exposure cap must not exceed EUR 10")
        if self.maximum_order_eur > self.maximum_total_exposure_eur:
            raise ValueError("approval order cap exceeds exposure cap")
        if self.maximum_open_positions != 1:
            raise ValueError("approval must allow exactly one open position")
        if self.maximum_new_orders_per_day != 1:
            raise ValueError("approval must allow exactly one new order per day")
        if self.autoscale or not self.spot_only:
            raise ValueError("approval must remain spot-only without autoscale")
        if self.approved_for_live and (not self.approval_reference or not self.approved_at):
            raise ValueError("live approval requires human approval evidence")


class LiveStrategyApprovalRegistry:
    """Read-only human authority registry; automation never writes this file."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def load(self) -> dict[str, LiveStrategyApproval]:
        if not self.path.is_file():
            return {}
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        if raw.get("default_policy") != "FAIL_CLOSED":
            raise ValueError("live approval registry is not fail-closed")
        strategies = raw.get("strategies")
        if not isinstance(strategies, dict):
            raise ValueError("live approval registry lacks strategies")
        result: dict[str, LiveStrategyApproval] = {}
        for strategy_id, values in strategies.items():
            if not isinstance(values, dict):
                raise ValueError("invalid live approval record")
            record = LiveStrategyApproval.from_mapping(
                str(strategy_id),
                values,
            )
            record.validate()
            result[record.strategy_id] = record
        return result

    def assess(
        self,
        strategy_id: str,
        strategy_dna_hash: str,
    ) -> tuple[LiveApprovalStatus, LiveStrategyApproval | None, str]:
        try:
            record = self.load().get(strategy_id)
        except (OSError, TypeError, ValueError, KeyError):
            return (
                LiveApprovalStatus.INVALID_POLICY,
                None,
                "LIVE_APPROVAL_REGISTRY_INVALID",
            )
        if record is None:
            return (
                LiveApprovalStatus.STRATEGY_MISSING,
                None,
                "LIVE_APPROVAL_STRATEGY_MISSING",
            )
        if record.strategy_dna_hash != strategy_dna_hash:
            return (
                LiveApprovalStatus.DNA_MISMATCH,
                record,
                "LIVE_APPROVAL_DNA_MISMATCH",
            )
        if not record.approved_for_live:
            return (
                LiveApprovalStatus.NOT_APPROVED,
                record,
                "LIVE_APPROVAL_HUMAN_CONFIRMATION_REQUIRED",
            )
        return (
            LiveApprovalStatus.APPROVED,
            record,
            "LIVE_APPROVAL_VERIFIED",
        )


@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    observed_at: str
    data_through: str | None
    primary_regime: MarketRegime
    active_regimes: tuple[MarketRegime, ...]
    confidence: float
    metrics: dict[str, float | None]
    reason_codes: tuple[str, ...]
    data_fresh: bool

    def payload(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "primary_regime": self.primary_regime.value,
            "active_regimes": [regime.value for regime in self.active_regimes],
        }


class MarketRegimeClassifier:
    """Classical closed-candle regime classifier with no learning loop."""

    @staticmethod
    def classify(
        frame: pd.DataFrame,
        *,
        observed_at: datetime | None = None,
        spread_bps: float | None = None,
    ) -> RegimeSnapshot:
        now = (observed_at or datetime.now(UTC)).astimezone(UTC)
        if frame.empty or "close" not in frame or len(frame) < 210:
            return RegimeSnapshot(
                observed_at=now.isoformat(),
                data_through=None,
                primary_regime=MarketRegime.UNCLASSIFIED,
                active_regimes=(MarketRegime.UNCLASSIFIED,),
                confidence=0.0,
                metrics={},
                reason_codes=("INSUFFICIENT_CLOSED_CANDLE_HISTORY",),
                data_fresh=False,
            )
        source = frame.copy()
        index = pd.to_datetime(source.index, utc=True)
        source.index = index
        if "closed" in source:
            source = source.loc[source["closed"].fillna(False).astype(bool)]
        close = pd.to_numeric(source["close"], errors="coerce").dropna()
        if len(close) < 210:
            return RegimeSnapshot(
                observed_at=now.isoformat(),
                data_through=(close.index[-1].isoformat() if not close.empty else None),
                primary_regime=MarketRegime.UNCLASSIFIED,
                active_regimes=(MarketRegime.UNCLASSIFIED,),
                confidence=0.0,
                metrics={},
                reason_codes=("INSUFFICIENT_CLOSED_CANDLE_HISTORY",),
                data_fresh=False,
            )
        latest_at = close.index[-1].to_pydatetime().astimezone(UTC)
        returns = np.log(close).diff()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        realized20 = returns.rolling(20).std(ddof=0) * math.sqrt(365.25)
        realized90 = returns.rolling(90).std(ddof=0) * math.sqrt(365.25)
        high20 = close.shift(1).rolling(20).max()
        mean20 = close.shift(1).rolling(20).mean()
        std20 = close.shift(1).rolling(20).std(ddof=0)
        zscore20 = (close - mean20) / std20.replace(0.0, np.nan)
        trend_strength = abs(float(ema50.iloc[-1] / ema200.iloc[-1] - 1.0))
        vol_ratio = float(
            realized20.iloc[-1] / realized90.iloc[-1] if realized90.iloc[-1] > 0 else 1.0
        )
        active: list[MarketRegime] = []
        reasons: list[str] = []
        liquidity_stress = bool(spread_bps is not None and spread_bps > 35.0)
        if liquidity_stress:
            active.append(MarketRegime.LIQUIDITY_STRESS)
            reasons.append("SPREAD_ABOVE_35_BPS")
        if close.iloc[-1] > high20.iloc[-1] and vol_ratio >= 1.15:
            structure = MarketRegime.BREAKOUT_EXPANSION
            reasons.append("CLOSE_ABOVE_PRIOR_20D_HIGH_WITH_VOL_EXPANSION")
        elif close.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1] and trend_strength >= 0.015:
            structure = MarketRegime.TREND_UP
            reasons.append("CLOSE_EMA50_EMA200_BULL_ALIGNMENT")
        elif close.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1] and trend_strength >= 0.015:
            structure = MarketRegime.TREND_DOWN
            reasons.append("CLOSE_EMA50_EMA200_BEAR_ALIGNMENT")
        elif abs(float(zscore20.iloc[-1])) >= 1.5:
            structure = MarketRegime.MEAN_REVERSION
            reasons.append("ABS_20D_ZSCORE_AT_LEAST_1_5")
        elif vol_ratio >= 1.15:
            structure = MarketRegime.RANGE_HIGH_VOL
            reasons.append("NON_TREND_VOLATILITY_EXPANSION")
        else:
            structure = MarketRegime.RANGE_LOW_VOL
            reasons.append("NON_TREND_NORMAL_OR_LOW_VOLATILITY")
        active.append(structure)
        risk = MarketRegime.RISK_ON if close.iloc[-1] > ema200.iloc[-1] else MarketRegime.RISK_OFF
        active.append(risk)
        fresh = latest_at >= now - timedelta(days=2)
        if not fresh:
            reasons.append("DAILY_DATA_STALE")
        confidence = min(
            100.0,
            55.0
            + min(25.0, trend_strength * 500.0)
            + min(20.0, abs(float(zscore20.iloc[-1])) * 5.0),
        )
        primary = MarketRegime.LIQUIDITY_STRESS if liquidity_stress else structure
        return RegimeSnapshot(
            observed_at=now.isoformat(),
            data_through=latest_at.isoformat(),
            primary_regime=primary,
            active_regimes=tuple(dict.fromkeys(active)),
            confidence=round(confidence, 4),
            metrics={
                "close": float(close.iloc[-1]),
                "ema50": float(ema50.iloc[-1]),
                "ema200": float(ema200.iloc[-1]),
                "realized_volatility_20d": float(realized20.iloc[-1]),
                "realized_volatility_90d": float(realized90.iloc[-1]),
                "volatility_ratio": vol_ratio,
                "zscore_20d": float(zscore20.iloc[-1]),
                "spread_bps": spread_bps,
            },
            reason_codes=tuple(reasons),
            data_fresh=fresh,
        )


@dataclass(frozen=True, slots=True)
class RoutedStrategy:
    strategy_id: str
    strategy_dna_hash: str
    mode: str
    eligible: bool
    regime_fit: float
    reason_codes: tuple[str, ...]


class StrategyRegimeRouter:
    """Route frozen strategies without changing their parameters."""

    _MAPPING: dict[str, tuple[MarketRegime, ...]] = {
        PRIMARY_STRATEGY_ID: (
            MarketRegime.MEAN_REVERSION,
            MarketRegime.RANGE_LOW_VOL,
            MarketRegime.RANGE_HIGH_VOL,
            MarketRegime.RISK_ON,
        ),
        "AMPS_P01_V90_T04": (
            MarketRegime.TREND_UP,
            MarketRegime.BREAKOUT_EXPANSION,
            MarketRegime.RISK_ON,
        ),
        "ROTATION_FROZEN_CONTROL": (
            MarketRegime.TREND_UP,
            MarketRegime.RISK_ON,
        ),
    }

    def route(
        self,
        regime: RegimeSnapshot,
    ) -> tuple[RoutedStrategy, ...]:
        active = set(regime.active_regimes)
        rows: list[RoutedStrategy] = []
        for strategy_id, favorable in self._MAPPING.items():
            matches = len(active.intersection(favorable))
            blocked = (
                not regime.data_fresh
                or MarketRegime.LIQUIDITY_STRESS in active
                or MarketRegime.RISK_OFF in active
            )
            rows.append(
                RoutedStrategy(
                    strategy_id=strategy_id,
                    strategy_dna_hash=(
                        PRIMARY_STRATEGY_DNA
                        if strategy_id == PRIMARY_STRATEGY_ID
                        else stable_hash(
                            ["FROZEN_SHADOW", strategy_id],
                            length=64,
                        )
                    ),
                    mode=(
                        "LIVE_CANARY_REVIEW"
                        if strategy_id == PRIMARY_STRATEGY_ID
                        else "FROZEN_SHADOW"
                    ),
                    eligible=matches > 0 and not blocked,
                    regime_fit=round(
                        min(1.0, matches / max(1, len(favorable) / 2)),
                        4,
                    ),
                    reason_codes=tuple(
                        [("REGIME_MATCH" if matches else "NO_FAVORABLE_REGIME_MATCH")]
                        + (["STALE_DATA_BLOCK"] if not regime.data_fresh else [])
                        + (
                            ["LIQUIDITY_STRESS_BLOCK"]
                            if MarketRegime.LIQUIDITY_STRESS in active
                            else []
                        )
                        + (["RISK_OFF_BLOCK"] if MarketRegime.RISK_OFF in active else [])
                    ),
                )
            )
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class Opportunity:
    opportunity_id: str
    strategy_id: str
    strategy_dna_hash: str
    market: str
    timeframe: str
    action: str
    confidence: float
    reward_risk: float
    regime_fit: float
    liquidity_score: float
    robustness_score: float
    score: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    valid_until: str
    blockers: tuple[str, ...]
    actionable: bool


class OpportunityScanner:
    """Rank natural signals; it never manufactures an entry."""

    @staticmethod
    def rank(
        opportunities: Sequence[Opportunity],
    ) -> tuple[Opportunity, ...]:
        return tuple(
            sorted(
                opportunities,
                key=lambda row: (
                    row.actionable,
                    row.score,
                    row.confidence,
                    row.reward_risk,
                    row.strategy_id,
                    row.market,
                ),
                reverse=True,
            )[:3]
        )

    @staticmethod
    def score(
        *,
        confidence: float,
        reward_risk: float,
        regime_fit: float,
        liquidity_score: float,
        robustness_score: float,
    ) -> float:
        return float(
            round(
                30.0 * np.clip(confidence / 100.0, 0.0, 1.0)
                + 20.0 * np.clip(reward_risk / 3.0, 0.0, 1.0)
                + 20.0 * np.clip(regime_fit, 0.0, 1.0)
                + 15.0 * np.clip(liquidity_score, 0.0, 1.0)
                + 15.0 * np.clip(robustness_score, 0.0, 1.0),
                4,
            )
        )


@dataclass(frozen=True, slots=True)
class ManagedPositionDecision:
    action: str
    reason_code: str
    quantity_fraction: float
    updated_stop_loss: float | None = None
    tp1_reached: bool | None = None


def decide_managed_position_action(
    position: Mapping[str, Any],
    *,
    market_price: float,
    strategy_action: str,
    owned_quantity: Decimal,
) -> ManagedPositionDecision:
    """Resolve one deterministic long-only position-management action."""

    if market_price <= 0:
        return ManagedPositionDecision(
            "BLOCK",
            "INVALID_PUBLIC_MARKET_PRICE",
            0.0,
        )
    if owned_quantity <= 0:
        return ManagedPositionDecision(
            "BLOCK",
            "MANAGED_POSITION_BALANCE_MISSING",
            0.0,
        )
    entry = float(position["entry_price"])
    stop = float(position["stop_loss"])
    tp1 = float(position["take_profit_1"])
    tp2 = float(position["take_profit_2"])
    tp1_seen = bool(position.get("tp1_reached"))
    if market_price <= stop:
        return ManagedPositionDecision(
            "SELL_FULL",
            "STOP_LOSS_REACHED",
            1.0,
            tp1_reached=tp1_seen,
        )
    if market_price >= tp2:
        return ManagedPositionDecision(
            "SELL_FULL",
            "TP2_REACHED",
            1.0,
            tp1_reached=True,
        )
    if market_price >= tp1 and not tp1_seen:
        # Keep the Level-1 position whole at TP1: fees, price movement and
        # quantity rounding could otherwise leave a sub-minimum remainder.
        # Record TP1 and protect the complete remainder at breakeven.
        return ManagedPositionDecision(
            "UPDATE_ONLY",
            "TP1_REACHED_MOVE_STOP_TO_BREAKEVEN",
            0.0,
            updated_stop_loss=max(stop, entry),
            tp1_reached=True,
        )
    if strategy_action == "EXIT":
        return ManagedPositionDecision(
            "SELL_FULL",
            "STRATEGY_EXIT",
            1.0,
            tp1_reached=tp1_seen,
        )
    return ManagedPositionDecision(
        "HOLD",
        "POSITION_WITHIN_PLAN",
        0.0,
        updated_stop_loss=stop,
        tp1_reached=tp1_seen,
    )


def _primary_parameters() -> ResidualReversalParameters:
    matches = [
        row
        for row in residual_reversal_parameter_set()
        if row.beta_lookback == 60 and row.residual_horizon == 5 and row.entry_zscore == -2.0
    ]
    if len(matches) != 1 or matches[0].dna_hash != PRIMARY_STRATEGY_DNA:
        raise RuntimeError("PRIMARY_STRATEGY_DNA_DRIFT")
    return matches[0]


def _load_primary_frames(
    settings: Settings,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], tuple[str, ...]]:
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    failures: list[str] = []
    for market in ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"):
        path = settings.paths.processed_data_dir / f"{market}_1d.parquet"
        if not path.is_file():
            failures.append(f"MISSING_DAILY_DATASET:{market}")
            continue
        frame = load_ohlcv(
            path,
            market=market,
            timeframe="1d",
            validate=False,
        )
        frames[market] = frame.sort_index()
        hashes[market] = sha256_file(path)
    return frames, hashes, tuple(failures)


def _closed_records_frame(records: Sequence[Any]) -> pd.DataFrame:
    """Convert provider records to a strictly closed, causal OHLCV frame."""

    rows = [
        {
            "timestamp": record.timestamp,
            "open": float(record.values["open"]),
            "high": float(record.values["high"]),
            "low": float(record.values["low"]),
            "close": float(record.values["close"]),
            "volume": float(record.values.get("volume") or 0.0),
        }
        for record in records
        if record.closed is True
    ]
    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
        )
    frame = pd.DataFrame(rows).set_index("timestamp")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame[~frame.index.duplicated(keep="last")].sort_index()


async def refresh_primary_daily_frames(
    settings: Settings,
    *,
    observed_at: datetime | None = None,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, str],
    tuple[str, ...],
    dict[str, Any],
]:
    """Fetch current public Bitvavo daily candles without private API access.

    The returned frames exist only in memory.  This keeps the reconciled
    research Parquet files immutable while ensuring live decisions never rely
    on a stale local snapshot.  Any missing market makes the execution path
    fail closed.
    """

    from data.data_loader import DataLoader

    now = (observed_at or datetime.now(UTC)).astimezone(UTC)
    loader = DataLoader(settings)
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    failures: list[str] = []
    market_provenance: dict[str, Any] = {}
    for market in ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"):
        try:
            records, provenance = await loader.download_canonical_ohlcv(
                provider="bitvavo",
                market=market,
                timeframe="1d",
                start=datetime(2018, 1, 1, tzinfo=UTC),
                end=now,
                run_id=f"live-public-daily-{now:%Y%m%dT%H%M%SZ}",
                resume=True,
                persist=False,
            )
            frame = _closed_records_frame(records)
            if len(frame) < 200:
                failures.append(f"INSUFFICIENT_FRESH_DAILY_DATA:{market}")
                continue
            frames[market] = frame
            hashes[market] = stable_hash(
                [record.raw_hash for record in records if record.closed is True],
                length=64,
            )
            market_provenance[market] = {
                **dict(provenance),
                "rows": len(frame),
                "data_through": frame.index[-1].isoformat(),
            }
        except Exception as exc:
            failures.append(
                f"PUBLIC_DAILY_REFRESH_FAILED:{market}:{type(exc).__name__}",
            )
    provenance_payload = {
        "provider": "bitvavo",
        "access": "PUBLIC_ONLY",
        "timeframe": "1d",
        "observed_at": now,
        "markets": market_provenance,
        "private_exchange_requests": 0,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    return frames, hashes, tuple(failures), provenance_payload


def _research_evidence(settings: Settings) -> dict[str, Any]:
    path = settings.paths.lab_dir / "reports" / "residual_reversal_campaign_v1.json"
    if not path.is_file():
        return {
            "status": "MISSING",
            "path": str(path),
            "strategy_match": False,
        }
    payload = read_json(path)
    primary = dict(payload.get("primary_result") or {})
    normal = dict(primary.get("normal") or {})
    metrics = dict(normal.get("metrics") or {})
    gates = dict(primary.get("gates") or {})
    economic_checks = dict(gates.get("economic_checks") or {})
    canary_core_checks = (
        "all_periods_positive",
        "all_stressed_periods_positive",
        "exposure_limits_respected",
        "maximum_drawdown",
        "minimum_effective_sample",
        "profit_factor",
        "stressed_validation_profit_factor",
        "strictly_prior_factor_causality",
        "validation_profit_factor",
    )
    return {
        "status": str(payload.get("status") or "UNKNOWN"),
        "path": str(path),
        "sha256": sha256_file(path),
        "strategy_id": payload.get("primary_strategy_id"),
        "strategy_dna_hash": primary.get("strategy_dna_hash"),
        "strategy_match": (
            payload.get("primary_strategy_id") == PRIMARY_STRATEGY_ID
            and primary.get("strategy_dna_hash") == PRIMARY_STRATEGY_DNA
        ),
        "holdout_status": payload.get("holdout_status"),
        "live_ready": bool(payload.get("live_ready")),
        "paper_candidates": int(payload.get("paper_candidates") or 0),
        "economic_pass": bool(gates.get("economic_pass")),
        "statistical_pass": bool(gates.get("statistical_pass")),
        "canary_economic_eligible": all(
            economic_checks.get(key) is True
            for key in canary_core_checks
        ),
        "economic_checks": economic_checks,
        "metrics": {
            "net_return": metrics.get("net_return"),
            "annualized_return": metrics.get("annualized_return"),
            "sharpe": metrics.get("sharpe"),
            "sortino": metrics.get("sortino"),
            "maximum_drawdown": metrics.get("maximum_drawdown"),
            "portfolio_period_profit_factor": metrics.get(
                "portfolio_period_profit_factor"
            ),
            "effective_sample_size": metrics.get(
                "portfolio_period_effective_sample_size"
            ),
            "average_exposure": metrics.get("average_exposure"),
        },
        "pbo": payload.get("pbo"),
    }


def _primary_opportunity(
    *,
    settings: Settings,
    frames: Mapping[str, pd.DataFrame],
    route: RoutedStrategy,
    evidence: Mapping[str, Any],
    regime: RegimeSnapshot,
    observed_at: datetime,
) -> tuple[Opportunity, dict[str, Any]]:
    policy = RotationPortfolioPolicy(
        allowed_markets=(
            "BTC-EUR",
            "ETH-EUR",
            "SOL-EUR",
            "LINK-EUR",
        ),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=200,
    )
    result = backtest_residual_reversal(
        frames,
        _primary_parameters(),
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        portfolio_policy=policy,
    )
    weights = result.executed_weights
    current = float(weights["ETH-EUR"].iloc[-1])
    previous = float(weights["ETH-EUR"].iloc[-2]) if len(weights) > 1 else 0.0
    if current > 0 and previous <= 0:
        action = "BUY"
    elif current <= 0 < previous:
        action = "EXIT"
    elif current > 0:
        action = "HOLD"
    else:
        action = "NO_SIGNAL"
    eth = frames["ETH-EUR"].copy()
    eth.index = pd.to_datetime(eth.index, utc=True)
    close = pd.to_numeric(eth["close"], errors="coerce")
    high = pd.to_numeric(eth["high"], errors="coerce")
    low = pd.to_numeric(eth["low"], errors="coerce")
    prior = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prior).abs(),
            (low - prior).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.rolling(14).mean().iloc[-1])
    entry = float(close.iloc[-1])
    risk_distance = max(atr * 2.0, entry * 0.03)
    stop = max(0.0, entry - risk_distance)
    tp1 = entry + risk_distance * 1.5
    tp2 = entry + risk_distance * 2.5
    confidence = min(
        90.0,
        55.0
        + 20.0 * route.regime_fit
        + (
            10.0
            if evidence.get("canary_economic_eligible")
            else 0.0
        ),
    )
    robustness = (
        0.70
        if evidence.get("canary_economic_eligible")
        else 0.35
    )
    reward_risk = 1.5
    score = OpportunityScanner.score(
        confidence=confidence,
        reward_risk=reward_risk,
        regime_fit=route.regime_fit,
        liquidity_score=0.90,
        robustness_score=robustness,
    )
    blockers: list[str] = []
    if action != "BUY":
        blockers.append("NO_NATURAL_NEW_ENTRY")
    if not route.eligible:
        blockers.extend(route.reason_codes)
    if not regime.data_fresh:
        blockers.append("STALE_DAILY_DATA")
    if not evidence.get("strategy_match"):
        blockers.append("RESEARCH_EVIDENCE_IDENTITY_MISMATCH")
    if not evidence.get("canary_economic_eligible"):
        blockers.append(
            "CANARY_CORE_ECONOMIC_CHECKS_NOT_PASSED"
        )
    opportunity_id = stable_hash(
        {
            "strategy_id": PRIMARY_STRATEGY_ID,
            "strategy_dna_hash": PRIMARY_STRATEGY_DNA,
            "market": "ETH-EUR",
            "timeframe": "1d",
            "action": action,
            "data_through": regime.data_through,
            "entry": round(entry, 8),
            "stop": round(stop, 8),
            "tp1": round(tp1, 8),
            "tp2": round(tp2, 8),
        },
        length=32,
    )
    return (
        Opportunity(
            opportunity_id=opportunity_id,
            strategy_id=PRIMARY_STRATEGY_ID,
            strategy_dna_hash=PRIMARY_STRATEGY_DNA,
            market="ETH-EUR",
            timeframe="1d",
            action=action,
            confidence=confidence,
            reward_risk=reward_risk,
            regime_fit=route.regime_fit,
            liquidity_score=0.90,
            robustness_score=robustness,
            score=score,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            valid_until=(observed_at + timedelta(days=1))
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .isoformat(),
            blockers=tuple(dict.fromkeys(blockers)),
            actionable=not blockers,
        ),
        {
            "latest_executed_weight": current,
            "previous_executed_weight": previous,
            "latest_decision": (
                result.decisions.iloc[-1].to_dict() if not result.decisions.empty else None
            ),
            "integrity": result.integrity,
            "engine_version": RESIDUAL_REVERSAL_ENGINE_VERSION,
        },
    )


def build_autonomous_control_plane(
    settings: Settings,
    *,
    observed_at: datetime | None = None,
    write_artifacts: bool = True,
    frames_override: Mapping[str, pd.DataFrame] | None = None,
    data_hashes_override: Mapping[str, str] | None = None,
    data_failures_override: Sequence[str] = (),
    data_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic control-plane artifacts without private API calls."""

    now = (observed_at or datetime.now(UTC)).astimezone(UTC)
    output = settings.paths.output_dir / "reports"
    if frames_override is None:
        frames, data_hashes, data_failures = _load_primary_frames(settings)
        selected_data_provenance: dict[str, Any] = {
            "source": "LOCAL_RECONCILED_PARQUET",
            "access": "FILESYSTEM",
        }
    else:
        frames = {
            market: frame.copy()
            for market, frame in frames_override.items()
        }
        data_hashes = dict(data_hashes_override or {})
        data_failures = tuple(data_failures_override)
        selected_data_provenance = dict(data_provenance or {})
    btc = frames.get("BTC-EUR", pd.DataFrame())
    regime = MarketRegimeClassifier.classify(btc, observed_at=now)
    routes = StrategyRegimeRouter().route(regime)
    primary_route = next(row for row in routes if row.strategy_id == PRIMARY_STRATEGY_ID)
    evidence = _research_evidence(settings)
    failures = list(data_failures)
    opportunity: Opportunity | None = None
    diagnostics: dict[str, Any] = {}
    if not failures and evidence.get("strategy_match"):
        try:
            opportunity, diagnostics = _primary_opportunity(
                settings=settings,
                frames=frames,
                route=primary_route,
                evidence=evidence,
                regime=regime,
                observed_at=now,
            )
        except Exception as exc:
            failures.append(f"PRIMARY_SIGNAL_EVALUATION_FAILED:{type(exc).__name__}")
    else:
        failures.append("PRIMARY_SIGNAL_NOT_EVALUATED")
    ranked = OpportunityScanner.rank([opportunity] if opportunity is not None else [])
    registry_path = settings.paths.project_root / "config" / "live_strategy_approvals.yaml"
    registry = LiveStrategyApprovalRegistry(registry_path)
    approval_status, approval, approval_reason = registry.assess(
        PRIMARY_STRATEGY_ID,
        PRIMARY_STRATEGY_DNA,
    )
    from core.practical_governance import (
        capital_scaling_status_from_ledger,
        live_canary_authority,
    )

    operator_authorized, operator_authority, operator_authority_failures = (
        live_canary_authority(
            settings.paths.project_root,
            strategy_id=PRIMARY_STRATEGY_ID,
            strategy_dna=PRIMARY_STRATEGY_DNA,
            market="ETH-EUR",
        )
    )
    capital_scaling = capital_scaling_status_from_ledger(
        settings.paths.project_root,
        strategy_id=PRIMARY_STRATEGY_ID,
    )
    live_failures = list(settings.static_live_preflight_failures())
    if operator_authorized:
        overridable = {
            "LIVE_BLOCKED_NOT_PRODUCTION",
            "LIVE_BLOCKED_MODE_NOT_LIVE",
            "LIVE_BLOCKED_DISABLED",
            "LIVE_BLOCKED_MANUAL_APPROVAL",
            "LIVE_BLOCKED_CANARY_DISABLED",
        }
        live_failures = [
            reason for reason in live_failures if reason not in overridable
        ]
    else:
        live_failures.extend(operator_authority_failures)
    if approval_status is not LiveApprovalStatus.APPROVED:
        live_failures.append(approval_reason)
    if not ranked or not ranked[0].actionable:
        live_failures.append("NO_ACTIONABLE_NATURAL_OPPORTUNITY")
    if failures:
        live_failures.extend(failures)
    live_failures = list(dict.fromkeys(live_failures))
    context_checkpoint_path = (
        settings.paths.checkpoints_dir
        / "prospective_context_hourly.json"
    )
    context_checkpoint = (
        dict(read_json(context_checkpoint_path))
        if context_checkpoint_path.is_file()
        else {}
    )
    regime_payload = {
        "schema_version": "current_regime_v1",
        **regime.payload(),
        "orders_generated": 0,
    }
    router_payload = {
        "schema_version": "strategy_regime_router_v1",
        "observed_at": now,
        "routes": [
            {
                **asdict(row),
                "reason_codes": list(row.reason_codes),
            }
            for row in routes
        ],
        "parameters_changed": 0,
        "automatic_live_promotions": 0,
        "orders_generated": 0,
    }
    opportunities_payload = {
        "schema_version": "top_opportunities_v1",
        "observed_at": now,
        "top_opportunities": [
            {
                **asdict(row),
                "blockers": list(row.blockers),
            }
            for row in ranked
        ],
        "eligible_count": sum(row.actionable for row in ranked),
        "selected_for_execution": (asdict(ranked[0]) if ranked and ranked[0].actionable else None),
        "maximum_executions_this_cycle": 1,
        "orders_generated": 0,
    }
    approval_payload = (
        {
            **asdict(approval),
            "registry_sha256": sha256_file(registry_path),
        }
        if approval is not None
        else None
    )
    live_payload = {
        "schema_version": "live_trading_status_v1",
        "observed_at": now,
        "status": "READY" if not live_failures else "BLOCKED",
        "strategy_id": PRIMARY_STRATEGY_ID,
        "strategy_dna_hash": PRIMARY_STRATEGY_DNA,
        "research_evidence": evidence,
        "approval_status": approval_status.value,
        "approval_reason": approval_reason,
        "approval_policy": approval_payload,
        "operator_canary_authorized": operator_authorized,
        "operator_authority": operator_authority,
        "live_preflight_failures": live_failures,
        "natural_signal": (asdict(ranked[0]) if ranked else None),
        "signal_diagnostics": diagnostics,
        "data_hashes": data_hashes,
        "data_provenance": selected_data_provenance,
        "capital_scaling": capital_scaling,
        "canary_limits": {
            "capital_level": capital_scaling["active_level"],
            "maximum_order_eur": capital_scaling["caps"].get("max_order_eur"),
            "maximum_total_exposure_eur": capital_scaling["caps"].get(
                "max_exposure_eur"
            ),
            "maximum_exposure_pct": capital_scaling["caps"].get(
                "max_exposure_pct"
            ),
            "maximum_open_positions": capital_scaling["caps"]["max_positions"],
            "maximum_new_orders_per_day": 1,
            "spot_only": settings.execution.spot_only,
            "autoscale": False,
        },
        "private_exchange_requests": 0,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    current_position_path = output / "current_position.json"
    existing_position = (
        dict(read_json(current_position_path))
        if current_position_path.is_file()
        else {}
    )
    position_payload = (
        existing_position
        if existing_position.get("status")
        in {
            "OPEN",
            "OPEN_PENDING_RECONCILIATION",
            "PARTIALLY_REDUCED",
            "EXIT_PENDING_RECONCILIATION",
        }
        else {
            "schema_version": "current_position_v1",
            "status": "NOT_RECONCILED_PRIVATE_READ_NOT_REQUESTED",
            "position": None,
            "orders_generated": 0,
        }
    )
    artifacts = {
        "current_regime.json": regime_payload,
        "strategy_router_status.json": router_payload,
        "top_opportunities.json": opportunities_payload,
        "live_trading_status.json": live_payload,
        "current_position.json": position_payload,
        "reconciliation_status.json": {
            "schema_version": "reconciliation_status_v1",
            "status": "NOT_RUN_PRIVATE_READ_NOT_REQUESTED",
            "healthy": None,
            "unknown_orders": None,
            "orders_generated": 0,
        },
        "autopilot_status.json": {
            "schema_version": "autopilot_status_v1",
            "status": "RESEARCH_ONLY_ORDERLESS",
            "top_50_tracking_required": True,
            "top_50_tracking_status": (
                "PASSED"
                if context_checkpoint.get("status") == "PASSED"
                and int(context_checkpoint.get("ranking_count") or 0)
                == 50
                else "BLOCKED"
            ),
            "top_50_ranking_count": int(
                context_checkpoint.get("ranking_count") or 0
            ),
            "top_50_snapshot_hash": context_checkpoint.get(
                "snapshot_hash"
            ),
            "top_50_last_completed_epoch": context_checkpoint.get(
                "last_completed_epoch"
            ),
            "automatic_live_promotions": 0,
            "live_registry_mutations": 0,
            "orders_generated": 0,
        },
        "daily_execution_summary.json": {
            "schema_version": "daily_execution_summary_v1",
            "date": now.date().isoformat(),
            "natural_opportunities": len(ranked),
            "actionable_opportunities": sum(row.actionable for row in ranked),
            "orders_generated": 0,
            "orders_submitted": 0,
            "reason": (
                "PREFLIGHT_PASSED_NO_SUBMISSION_REQUESTED"
                if not live_failures
                else "LIVE_FAIL_CLOSED"
            ),
        },
    }
    if write_artifacts:
        output.mkdir(parents=True, exist_ok=True)
        for name, payload in artifacts.items():
            atomic_write_json(output / name, payload)
    return {
        "status": live_payload["status"],
        "observed_at": now,
        "regime": regime_payload,
        "router": router_payload,
        "opportunities": opportunities_payload,
        "live": live_payload,
        "artifacts": {name: str(output / name) for name in artifacts},
        "orders_generated": 0,
        "orders_submitted": 0,
    }


async def build_fresh_autonomous_control_plane(
    settings: Settings,
    *,
    observed_at: datetime | None = None,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Build the live control plane from current public closed candles."""

    now = (observed_at or datetime.now(UTC)).astimezone(UTC)
    frames, hashes, failures, provenance = (
        await refresh_primary_daily_frames(
            settings,
            observed_at=now,
        )
    )
    return build_autonomous_control_plane(
        settings,
        observed_at=now,
        write_artifacts=write_artifacts,
        frames_override=frames,
        data_hashes_override=hashes,
        data_failures_override=failures,
        data_provenance=provenance,
    )


def _notify_autonomous_event_safely(
    settings: Settings,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from notifications.telegram import TelegramNotifier

        return TelegramNotifier(
            settings.telegram,
            output_directory=(
                settings.paths.output_dir / "notifications"
            ),
            allowed_markets=settings.operational.markets,
        ).notify_system_event(event_type, payload)
    except Exception as exc:
        return {
            "delivery_status": "FAILED_ISOLATED",
            "reason_code": f"TELEGRAM_{type(exc).__name__.upper()}",
            "orders_generated": 0,
            "orders_submitted": 0,
        }


async def _bitvavo_public_price(
    session: Any,
    market: str,
) -> Decimal:
    import aiohttp

    try:
        async with session.get(
            "https://api.bitvavo.com/v2/ticker/price",
            params={"market": market},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"PUBLIC_PRICE_HTTP_{response.status}"
                )
            payload = await response.json(content_type=None)
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise RuntimeError("PUBLIC_PRICE_UNAVAILABLE") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("PUBLIC_PRICE_INVALID_RESPONSE")
    price = Decimal(str(payload.get("price") or "0"))
    if price <= 0:
        raise RuntimeError("PUBLIC_PRICE_NON_POSITIVE")
    return price


async def _bitvavo_entry_liquidity(
    session: Any,
    *,
    market: str,
    requested_notional_eur: Decimal,
    settings: Settings,
) -> dict[str, Any]:
    """Fail closed on entry when current public liquidity cannot support it."""

    import aiohttp

    limits = settings.autonomous_live.liquidity_limits(market)
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with session.get(
            f"https://api.bitvavo.com/v2/{market}/book",
            params={"depth": 100},
            timeout=timeout,
        ) as response:
            if response.status >= 400:
                raise RuntimeError(f"PUBLIC_BOOK_HTTP_{response.status}")
            book = await response.json(content_type=None)
        async with session.get(
            "https://api.bitvavo.com/v2/ticker/24h",
            params={"market": market},
            timeout=timeout,
        ) as response:
            if response.status >= 400:
                raise RuntimeError(f"PUBLIC_TICKER_HTTP_{response.status}")
            ticker = await response.json(content_type=None)
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise RuntimeError("PUBLIC_LIQUIDITY_UNAVAILABLE") from exc
    if not isinstance(book, dict) or not isinstance(ticker, dict):
        raise RuntimeError("PUBLIC_LIQUIDITY_INVALID_RESPONSE")
    bids = [
        (Decimal(str(row[0])), Decimal(str(row[1])))
        for row in book.get("bids", [])
        if isinstance(row, (list, tuple)) and len(row) >= 2
    ]
    asks = [
        (Decimal(str(row[0])), Decimal(str(row[1])))
        for row in book.get("asks", [])
        if isinstance(row, (list, tuple)) and len(row) >= 2
    ]
    bids = [(price, amount) for price, amount in bids if price > 0 and amount > 0]
    asks = [(price, amount) for price, amount in asks if price > 0 and amount > 0]
    if not bids or not asks:
        raise RuntimeError("PUBLIC_ORDERBOOK_EMPTY")
    best_bid = max(price for price, _ in bids)
    best_ask = min(price for price, _ in asks)
    mid = (best_bid + best_ask) / Decimal("2")
    if mid <= 0 or best_ask <= best_bid:
        raise RuntimeError("PUBLIC_ORDERBOOK_CROSSED_OR_INVALID")
    spread_bps = (best_ask - best_bid) / mid * Decimal("10000")
    visible_ask_depth_eur = sum(price * amount for price, amount in asks)
    last = Decimal(str(ticker.get("last") or best_ask))
    base_volume = Decimal(str(ticker.get("volume") or "0"))
    quote_volume = Decimal(
        str(
            ticker.get("volumeQuote")
            or ticker.get("quoteVolume")
            or (base_volume * last)
        )
    )
    remaining = requested_notional_eur
    acquired = Decimal("0")
    spent = Decimal("0")
    for price, amount in sorted(asks):
        if remaining <= 0:
            break
        level_notional = price * amount
        take_notional = min(remaining, level_notional)
        acquired += take_notional / price
        spent += take_notional
        remaining -= take_notional
    estimated_average_price = spent / acquired if acquired > 0 else Decimal("0")
    estimated_slippage_bps = (
        (estimated_average_price / best_ask - Decimal("1")) * Decimal("10000")
        if estimated_average_price > 0
        else Decimal("Infinity")
    )
    participation_pct = (
        requested_notional_eur / visible_ask_depth_eur * Decimal("100")
        if visible_ask_depth_eur > 0
        else Decimal("Infinity")
    )
    blockers: list[str] = []
    if remaining > 0:
        blockers.append("INSUFFICIENT_VISIBLE_ASK_LIQUIDITY")
    if spread_bps > Decimal(str(limits["maximum_spread_bps"])):
        blockers.append("SPREAD_LIMIT_EXCEEDED")
    if visible_ask_depth_eur < Decimal(
        str(limits["minimum_visible_ask_depth_eur"])
    ):
        blockers.append("MINIMUM_VISIBLE_ASK_DEPTH_NOT_MET")
    if quote_volume < Decimal(str(limits["minimum_24h_quote_volume_eur"])):
        blockers.append("MINIMUM_24H_QUOTE_VOLUME_NOT_MET")
    if estimated_slippage_bps > Decimal(
        str(limits["maximum_slippage_bps"])
    ):
        blockers.append("ESTIMATED_SLIPPAGE_LIMIT_EXCEEDED")
    if participation_pct > Decimal(
        str(limits["maximum_visible_liquidity_participation_pct"])
    ):
        blockers.append("VISIBLE_LIQUIDITY_PARTICIPATION_LIMIT_EXCEEDED")
    return {
        "status": "PASSED" if not blockers else "BLOCKED",
        "market": market,
        "requested_notional_eur": str(requested_notional_eur),
        "best_bid": str(best_bid),
        "best_ask": str(best_ask),
        "spread_bps": str(spread_bps),
        "visible_ask_depth_eur": str(visible_ask_depth_eur),
        "quote_volume_24h_eur": str(quote_volume),
        "estimated_average_price": str(estimated_average_price),
        "estimated_slippage_bps": str(estimated_slippage_bps),
        "visible_liquidity_participation_pct": str(participation_pct),
        "limits": limits,
        "blocking_reasons": blockers,
    }


# Public internal-service adapters.  Generated live strategies reuse the same
# public market reads and sanitized notification path as the RR canary.
bitvavo_public_price = _bitvavo_public_price
bitvavo_entry_liquidity = _bitvavo_entry_liquidity
notify_autonomous_event_safely = _notify_autonomous_event_safely


async def execute_autonomous_canary_once(
    settings: Settings,
    *,
    submit: bool,
    force_exit: bool = False,
    allow_new_entry: bool = True,
) -> dict[str, Any]:
    """Run one private reconciliation and, when authorized, one canary buy.

    This function performs no private exchange request while the static,
    human-approval, evidence, regime or natural-signal gates are blocked.
    """

    result = await build_fresh_autonomous_control_plane(settings)
    live = dict(result["live"])
    position_path = (
        settings.paths.output_dir
        / "reports"
        / "current_position.json"
    )
    position_artifact = (
        dict(read_json(position_path))
        if position_path.is_file()
        else {}
    )
    position = dict(position_artifact.get("position") or {})
    position_status = str(position_artifact.get("status") or "")
    managed_position = bool(
        position
        and position_status
        in {
            "OPEN",
            "OPEN_PENDING_RECONCILIATION",
            "PARTIALLY_REDUCED",
            "EXIT_PENDING_RECONCILIATION",
        }
        and position.get("strategy_id") == PRIMARY_STRATEGY_ID
        and position.get("strategy_dna_hash")
        == PRIMARY_STRATEGY_DNA
    )
    if force_exit and not managed_position:
        return {
            **live,
            "status": "BLOCKED",
            "cycle_status": "NO_TRADE",
            "reason_code": "NO_MANAGED_POSITION_TO_CLOSE",
            "private_exchange_requests": 0,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    entry_only_failures = {
        "NO_ACTIONABLE_NATURAL_OPPORTUNITY",
    }
    authority_failures = [
        reason
        for reason in live["live_preflight_failures"]
        if reason not in entry_only_failures
    ]
    if authority_failures:
        return {
            **live,
            "cycle_status": "NO_TRADE",
            "reason_code": "LIVE_PREFLIGHT_BLOCKED",
            "authority_failures": authority_failures,
            "private_exchange_requests": 0,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    opportunity = dict(live.get("natural_signal") or {})
    natural_buy = bool(
        opportunity.get("actionable")
        and opportunity.get("action") == "BUY"
    )
    if not managed_position and not natural_buy:
        return {
            **live,
            "cycle_status": "NO_TRADE",
            "reason_code": "NO_ACTIONABLE_NATURAL_BUY",
            "private_exchange_requests": 0,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    if not managed_position and not allow_new_entry:
        return {
            **live,
            "status": "BLOCKED",
            "cycle_status": "NO_TRADE",
            "reason_code": "NEW_ENTRIES_BLOCKED_BY_OPERATIONAL_STATE",
            "private_exchange_requests": 0,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    now = datetime.now(UTC)
    if not managed_position and (
        now.hour != 0 or now.minute > 15
    ):
        return {
            **live,
            "status": "BLOCKED",
            "cycle_status": "NO_TRADE",
            "reason_code": "NEXT_DAILY_OPEN_EXECUTION_WINDOW_CLOSED",
            "private_exchange_requests": 0,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    if not submit:
        return {
            **live,
            "cycle_status": "READY_NOT_SUBMITTED",
            "reason_code": "PREFLIGHT_ONLY",
            "private_exchange_requests": 0,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    import aiohttp

    from core.contracts import (
        ExecutionBlocked,
        OrderIntent,
        OrderSide,
        OrderType,
        ReconciliationRequired,
        ResearchStatus,
    )
    from core.economics import CanonicalCostModel
    from execution.execution import LivePreflight, build_live_client
    from portfolio.buy_chain import (
        canonicalize_approved_buy_order,
        planned_target_net_edge,
    )
    from risk.canary_guard import CanaryPolicy, InstitutionalCanaryGuard
    from risk.risk_manager import (
        KillSwitch,
        PortfolioSnapshot,
        RiskManager,
    )

    ledger_path = settings.paths.checkpoints_dir / "live_execution.jsonl"
    market = str(
        position.get("market")
        if managed_position
        else opportunity["market"]
    )
    estimated_price = Decimal(
        str(
            position.get("entry_price")
            if managed_position
            else opportunity["entry_price"]
        )
    )
    capital_scaling = dict(live.get("capital_scaling") or {})
    capital_level = int(capital_scaling.get("active_level") or 1)
    capital_caps = dict(capital_scaling.get("caps") or {})
    requested_notional = Decimal(
        str(
            capital_caps.get("max_order_eur")
            or settings.execution.maximum_live_order_eur
        )
    )
    effective_cap_limits: dict[str, Any] = {
        "capital_level": capital_level,
        "max_order_eur": float(requested_notional),
        "max_exposure_eur": float(
            capital_caps.get("max_exposure_eur")
            or settings.execution.maximum_live_total_eur
        ),
        "max_positions": int(
            capital_caps.get("max_positions")
            or settings.execution.maximum_live_open_positions
        ),
        "max_new_orders_per_day": 1,
    }
    quantity = requested_notional / estimated_price
    kill_switch = KillSwitch(settings.paths.checkpoints_dir / "kill_switch.json")
    if kill_switch.active and not managed_position:
        return {
            **live,
            "status": "BLOCKED",
            "cycle_status": "NO_TRADE",
            "reason_code": "LIVE_BLOCKED_KILL_SWITCH",
            "private_exchange_requests": 0,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    if kill_switch.active and managed_position:
        force_exit = True
    async with aiohttp.ClientSession() as session:
        client = build_live_client(
            settings,
            session=session,
            ledger_path=ledger_path,
        )
        balances = await client.balances()
        by_symbol = {str(item.get("symbol")): item for item in balances}
        eur = Decimal(str(by_symbol.get("EUR", {}).get("available", "0")))
        base = market.split("-")[0]
        from core.account_inventory import (
            load_inventory_baseline,
            reconcile_inventory,
        )

        inventory_baseline, inventory_failures = (
            load_inventory_baseline(
                settings,
                authority=dict(live.get("operator_authority") or {}),
            )
        )
        if inventory_failures:
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "NO_TRADE",
                "reason_code": "PREEXISTING_INVENTORY_BASELINE_INVALID",
                "inventory_failures": list(inventory_failures),
                "private_exchange_requests": 1,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        inventory_reconciliation = reconcile_inventory(
            balances,
            inventory_baseline,
        )
        excess_by_symbol = {
            symbol: Decimal(str(quantity))
            for symbol, quantity in dict(
                inventory_reconciliation["excess"]
            ).items()
        }
        owned = excess_by_symbol.get(base, Decimal("0"))
        unexpected_non_eur = set(excess_by_symbol)
        if managed_position:
            unexpected_non_eur.discard(base)
        if managed_position and position_status in {
            "OPEN_PENDING_RECONCILIATION",
            "EXIT_PENDING_RECONCILIATION",
        }:
            client_order_id = str(
                position.get(
                    "entry_client_order_id"
                    if position_status
                    == "OPEN_PENDING_RECONCILIATION"
                    else "exit_client_order_id"
                )
                or ""
            )
            if not client_order_id:
                return {
                    **live,
                    "status": "BLOCKED",
                    "cycle_status": "NO_TRADE",
                    "reason_code": (
                        "PENDING_POSITION_MISSING_CLIENT_ORDER_ID"
                    ),
                    "private_exchange_requests": 1,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
            try:
                pending_order = await client.get_order(
                    market=market,
                    client_order_id=client_order_id,
                )
            except (ExecutionBlocked, ReconciliationRequired):
                return {
                    **live,
                    "status": "BLOCKED",
                    "cycle_status": "RECONCILIATION_REQUIRED",
                    "reason_code": "PENDING_ORDER_RECONCILIATION_FAILED",
                    "private_exchange_requests": 2,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
            pending_status = (
                str(pending_order.get("status") or "")
                .replace("_", "")
                .replace("-", "")
                .casefold()
            )
            pending_states = {
                "new",
                "awaitingtrigger",
                "partiallyfilled",
                "pending",
            }
            failed_states = {
                "canceled",
                "cancelled",
                "expired",
                "rejected",
            }
            if pending_status in pending_states:
                return {
                    **live,
                    "status": "POSITION_RECONCILIATION_PENDING",
                    "cycle_status": "NO_NEW_ORDER",
                    "reason_code": (
                        "ENTRY_ORDER_STILL_PENDING"
                        if position_status
                        == "OPEN_PENDING_RECONCILIATION"
                        else "EXIT_ORDER_STILL_PENDING"
                    ),
                    "order_status": pending_order.get("status"),
                    "private_exchange_requests": 2,
                    "orders_generated": 0,
                    "orders_submitted": 0,
            }
            if pending_status == "filled":
                client.record_final_fill(
                    pending_order,
                    fallback_market=market,
                    fallback_side=(
                        OrderSide.BUY
                        if position_status == "OPEN_PENDING_RECONCILIATION"
                        else OrderSide.SELL
                    ),
                    fallback_quantity=Decimal(
                        str(position.get("quantity") or owned)
                    ),
                    fallback_price=Decimal(
                        str(
                            pending_order.get("price")
                            or position.get("entry_price")
                            or estimated_price
                        )
                    ),
                )
                if position_status == "EXIT_PENDING_RECONCILIATION":
                    closed = {
                        **position,
                        "quantity": "0",
                        "closed_at": now,
                        "last_reconciled_at": now,
                        "exit_status": pending_order.get("status"),
                    }
                    atomic_write_json(
                        position_path,
                        {
                            "schema_version": "current_position_v1",
                            "status": "CLOSED",
                            "position": closed,
                            "orders_generated": 0,
                            "orders_submitted": 0,
                        },
                    )
                    notification = (
                        _notify_autonomous_event_safely(
                            settings,
                            "ORDER_FILLED",
                            {
                                "order_id": pending_order.get(
                                    "orderId"
                                ),
                                "market": market,
                                "side": "SELL",
                                "order_type": (
                                    pending_order.get("orderType")
                                    or "MARKET"
                                ),
                                "price": pending_order.get(
                                    "filledAmountQuote"
                                )
                                or pending_order.get("price"),
                                "requested_quantity": position.get(
                                    "quantity"
                                ),
                                "filled_quantity": pending_order.get(
                                    "filledAmount"
                                )
                                or position.get("quantity"),
                                "remaining_quantity": pending_order.get(
                                    "amountRemaining"
                                )
                                or "0",
                                "average_fill_price": pending_order.get(
                                    "price"
                                ),
                                "invested_eur": pending_order.get(
                                    "filledAmountQuote"
                                ),
                                "fee": pending_order.get("feePaid"),
                                "strategy_id": (
                                    PRIMARY_STRATEGY_ID
                                ),
                                "timeframe": PRIMARY_TIMEFRAME,
                                "venue_timestamp": pending_order.get(
                                    "updated"
                                ),
                                "status": pending_order.get(
                                    "status"
                                ),
                                "verification_source": (
                                    "BITVAVO_REST_RECONCILIATION"
                                ),
                            },
                        )
                    )
                    return {
                        **live,
                        "status": "POSITION_CLOSED",
                        "cycle_status": "RECONCILED",
                        "reason_code": "EXIT_ORDER_FILLED_RECONCILED",
                        "notification": notification,
                        "private_exchange_requests": 2,
                        "orders_generated": 0,
                        "orders_submitted": 0,
                    }
                opened = {
                    **position,
                    "quantity": str(
                        pending_order.get("filledAmount")
                        or owned
                    ),
                    "entry_price": float(
                        pending_order.get("price")
                        or position.get("entry_price")
                    ),
                    "last_reconciled_at": now,
                    "entry_status": pending_order.get("status"),
                }
                atomic_write_json(
                    position_path,
                    {
                        "schema_version": "current_position_v1",
                        "status": "OPEN",
                        "position": opened,
                        "orders_generated": 0,
                        "orders_submitted": 0,
                    },
                )
                notification = _notify_autonomous_event_safely(
                    settings,
                    "ORDER_FILLED",
                    {
                        "order_id": pending_order.get("orderId"),
                        "market": market,
                        "side": "BUY",
                        "order_type": pending_order.get("orderType") or "MARKET",
                        "requested_quantity": position.get("quantity"),
                        "filled_quantity": pending_order.get("filledAmount") or owned,
                        "remaining_quantity": pending_order.get("amountRemaining") or "0",
                        "average_fill_price": pending_order.get("price")
                        or position.get("entry_price"),
                        "invested_eur": pending_order.get("filledAmountQuote"),
                        "fee": pending_order.get("feePaid"),
                        "strategy_id": PRIMARY_STRATEGY_ID,
                        "timeframe": PRIMARY_TIMEFRAME,
                        "venue_timestamp": pending_order.get("updated"),
                        "status": pending_order.get("status"),
                        "verification_source": "BITVAVO_REST_RECONCILIATION",
                    },
                )
                return {
                    **live,
                    "status": "POSITION_OPEN",
                    "cycle_status": "RECONCILED",
                    "reason_code": "ENTRY_ORDER_FILLED_RECONCILED",
                    "notification": notification,
                    "private_exchange_requests": 2,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
            if pending_status in failed_states:
                restored_status = (
                    "OPEN"
                    if position_status
                    == "EXIT_PENDING_RECONCILIATION"
                    and owned > 0
                    else "CLOSED"
                )
                atomic_write_json(
                    position_path,
                    {
                        "schema_version": "current_position_v1",
                        "status": restored_status,
                        "position": {
                            **position,
                            "quantity": str(owned),
                            "last_reconciled_at": now,
                            "order_failure_status": (
                                pending_order.get("status")
                            ),
                        },
                        "orders_generated": 0,
                        "orders_submitted": 0,
                    },
                )
                notification = _notify_autonomous_event_safely(
                    settings,
                    "ORDER_REJECTED",
                    {
                        "order_id": pending_order.get("orderId"),
                        "market": market,
                        "side": (
                            "BUY"
                            if position_status
                            == "OPEN_PENDING_RECONCILIATION"
                            else "SELL"
                        ),
                        "order_type": (
                            pending_order.get("orderType")
                            or "MARKET"
                        ),
                        "strategy_id": PRIMARY_STRATEGY_ID,
                        "status": pending_order.get("status"),
                        "reason_code": (
                            "VENUE_ORDER_TERMINATED_WITHOUT_FILL"
                        ),
                    },
                )
                return {
                    **live,
                    "status": "BLOCKED",
                    "cycle_status": "ORDER_FAILED",
                    "reason_code": "VENUE_ORDER_TERMINATED_WITHOUT_FILL",
                    "notification": notification,
                    "private_exchange_requests": 2,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "RECONCILIATION_REQUIRED",
                "reason_code": "UNKNOWN_PENDING_ORDER_STATUS",
                "private_exchange_requests": 2,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        reconciliation = await client.reconcile(markets=(market,))
        if not reconciliation.healthy:
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "NO_TRADE",
                "reason_code": "LIVE_BLOCKED_RECONCILIATION",
                "reconciliation": asdict(reconciliation),
                "private_exchange_requests": 3,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        if managed_position and (owned <= 0 or unexpected_non_eur):
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "NO_TRADE",
                "reason_code": (
                    "MANAGED_POSITION_REMOTE_BALANCE_MISMATCH"
                ),
                "unexpected_non_eur_assets": sorted(
                    unexpected_non_eur
                ),
                "private_exchange_requests": 3,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        if not managed_position and excess_by_symbol:
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "NO_TRADE",
                "reason_code": "EXISTING_NON_EUR_POSITION_REQUIRES_RECONCILIATION",
                "unexpected_non_eur_assets": sorted(
                    excess_by_symbol
                ),
                "private_exchange_requests": 3,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        public_price = await _bitvavo_public_price(
            session,
            market,
        )
        live_equity = eur + owned * public_price
        if capital_level == 4:
            maximum_exposure_pct = Decimal(
                str(capital_caps.get("max_exposure_pct") or "0")
            )
            maximum_risk_pct = Decimal(
                str(capital_caps.get("max_risk_per_trade_pct") or "0")
            )
            maximum_total = (
                live_equity * maximum_exposure_pct / Decimal("100")
            )
            stop_price = Decimal(
                str(
                    position.get("stop_loss")
                    if managed_position
                    else opportunity.get("stop_loss")
                    or public_price
                )
            )
            stop_fraction = (
                abs(public_price - stop_price) / public_price
                if public_price > 0
                else Decimal("0")
            )
            risk_budget = live_equity * maximum_risk_pct / Decimal("100")
            risk_sized_notional = (
                risk_budget / stop_fraction
                if stop_fraction > 0
                else Decimal("0")
            )
            requested_notional = min(maximum_total, risk_sized_notional)
            effective_cap_limits = {
                "capital_level": 4,
                "account_equity_eur": float(live_equity),
                "max_order_eur": float(requested_notional),
                "max_exposure_eur": float(maximum_total),
                "max_exposure_pct": float(maximum_exposure_pct),
                "max_risk_per_trade_pct": float(maximum_risk_pct),
                "max_positions": int(capital_caps.get("max_positions") or 3),
                "max_new_orders_per_day": 1,
            }
        risk_state_path = (
            settings.paths.output_dir
            / "governance"
            / "live_canary_risk_state.json"
        )
        previous_risk_state = (
            dict(read_json(risk_state_path))
            if risk_state_path.is_file()
            else {}
        )
        today = now.date().isoformat()
        same_risk_day = previous_risk_state.get("date") == today
        raw_day_start_equity = Decimal(
            str(
                previous_risk_state.get(
                    "raw_day_start_equity_eur",
                    previous_risk_state.get("day_start_equity_eur"),
                )
                if same_risk_day
                else live_equity
            )
        )
        confirmed_capital_flow = confirmed_capital_flow_for_date(
            settings,
            today,
        )
        day_start_equity = raw_day_start_equity + confirmed_capital_flow
        peak_equity = max(
            live_equity,
            Decimal(str(previous_risk_state.get("peak_equity_eur") or live_equity)),
        )
        daily_loss_eur = max(Decimal("0"), day_start_equity - live_equity)
        drawdown_eur = max(Decimal("0"), peak_equity - live_equity)
        atomic_write_json(
            risk_state_path,
            {
                "schema_version": "live_canary_risk_state_v1",
                "date": today,
                "updated_at": now,
                "raw_day_start_equity_eur": str(raw_day_start_equity),
                "external_capital_flow_eur": str(confirmed_capital_flow),
                "day_start_equity_eur": str(day_start_equity),
                "peak_equity_eur": str(peak_equity),
                "current_equity_eur": str(live_equity),
                "daily_loss_eur": str(daily_loss_eur),
                "drawdown_eur": str(drawdown_eur),
                "maximum_daily_loss_eur": (
                    settings.execution.maximum_live_daily_loss_eur
                ),
                "maximum_drawdown_eur": (
                    settings.execution.maximum_live_drawdown_eur
                ),
            },
        )
        loss_limit_reached = daily_loss_eur >= Decimal(
            str(settings.execution.maximum_live_daily_loss_eur)
        )
        drawdown_limit_reached = drawdown_eur >= Decimal(
            str(settings.execution.maximum_live_drawdown_eur)
        )
        if (loss_limit_reached or drawdown_limit_reached) and not managed_position:
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "NO_TRADE",
                "reason_code": (
                    "DAILY_LOSS_LIMIT"
                    if loss_limit_reached
                    else "MAXIMUM_DRAWDOWN_LIMIT"
                ),
                "private_exchange_requests": 3,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        if loss_limit_reached or drawdown_limit_reached:
            force_exit = True
        if (
            not managed_position
            and abs(public_price / estimated_price - Decimal("1"))
            > Decimal("0.01")
        ):
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "NO_TRADE",
                "reason_code": "ENTRY_PRICE_DRIFT_ABOVE_ONE_PERCENT",
                "planned_entry_price": float(estimated_price),
                "public_market_price": float(public_price),
                "private_exchange_requests": 3,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        if not managed_position:
            estimated_price = public_price
            quantity = requested_notional / estimated_price
            try:
                liquidity = await _bitvavo_entry_liquidity(
                    session,
                    market=market,
                    requested_notional_eur=requested_notional,
                    settings=settings,
                )
            except RuntimeError as exc:
                return {
                    **live,
                    "status": "BLOCKED",
                    "cycle_status": "NO_TRADE",
                    "reason_code": str(exc),
                    "private_exchange_requests": 3,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
            live["entry_liquidity"] = liquidity
            if liquidity["status"] != "PASSED":
                return {
                    **live,
                    "status": "BLOCKED",
                    "cycle_status": "NO_TRADE",
                    "reason_code": "LIVE_BLOCKED_LIQUIDITY",
                    "liquidity_reason_codes": liquidity["blocking_reasons"],
                    "private_exchange_requests": 3,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
        live["effective_cap_limits"] = effective_cap_limits
        preflight = LivePreflight.evaluate(
            settings,
            markets=(market,),
            strategy_status=ResearchStatus.LIVE_BLOCKED,
            data_healthy=True,
            risk_manager_healthy=True,
            exchange_healthy=True,
            reconciliation_healthy=True,
            kill_switch_active=False,
            canary_exception_approved=True,
            operator_canary_authorized=bool(
                live.get("operator_canary_authorized")
            ),
            cap_limits=effective_cap_limits,
        )
        if not preflight.passed or preflight.capability is None:
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "NO_TRADE",
                "reason_code": "CANONICAL_LIVE_PREFLIGHT_FAILED",
                "canonical_failures": list(preflight.failures),
                "private_exchange_requests": 3,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        if managed_position:
            decision = decide_managed_position_action(
                position,
                market_price=float(public_price),
                strategy_action=str(
                    "EXIT"
                    if force_exit
                    else opportunity.get("action")
                    or "NO_SIGNAL"
                ),
                owned_quantity=owned,
            )
            reconciled_position = {
                **position,
                "quantity": str(owned),
                "last_market_price": float(public_price),
                "last_reconciled_at": now,
            }
            if decision.action in {"HOLD", "UPDATE_ONLY"}:
                if decision.updated_stop_loss is not None:
                    reconciled_position["stop_loss"] = (
                        decision.updated_stop_loss
                    )
                if decision.tp1_reached is not None:
                    reconciled_position["tp1_reached"] = (
                        decision.tp1_reached
                    )
                updated_artifact = {
                    "schema_version": "current_position_v1",
                    "status": "OPEN",
                    "position": reconciled_position,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
                atomic_write_json(position_path, updated_artifact)
                notification = None
                if decision.action == "UPDATE_ONLY":
                    notification = (
                        _notify_autonomous_event_safely(
                            settings,
                            "TP1_REACHED",
                            {
                                "market": market,
                                "status": "TP1_REACHED",
                                "price": float(public_price),
                                "strategy_id": (
                                    PRIMARY_STRATEGY_ID
                                ),
                                "reason_code": (
                                    decision.reason_code
                                ),
                            },
                        )
                    )
                return {
                    **live,
                    "status": "POSITION_MANAGED",
                    "cycle_status": decision.action,
                    "reason_code": decision.reason_code,
                    "position": reconciled_position,
                    "notification": notification,
                    "private_exchange_requests": 3,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
            if decision.action != "SELL_FULL":
                return {
                    **live,
                    "status": "BLOCKED",
                    "cycle_status": "NO_TRADE",
                    "reason_code": decision.reason_code,
                    "private_exchange_requests": 3,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
            exit_intent = OrderIntent(
                intent_id=stable_hash(
                    [
                        "AUTONOMOUS_CANARY_EXIT",
                        position.get("entry_opportunity_id"),
                        decision.reason_code,
                    ],
                    length=32,
                ),
                idempotency_key=(
                    "autonomous-canary-exit:"
                    f"{position.get('entry_opportunity_id')}:"
                    f"{decision.reason_code}"
                ),
                market=market,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=owned,
                strategy_id=PRIMARY_STRATEGY_ID,
                strategy_dna_hash=PRIMARY_STRATEGY_DNA,
                signal_id=str(
                    position.get("entry_opportunity_id") or ""
                )
                or None,
                portfolio_decision_id=stable_hash(
                    [
                        "LIVE_EXIT_DECISION",
                        PRIMARY_STRATEGY_DNA,
                        market,
                        position.get("entry_opportunity_id"),
                        decision.reason_code,
                    ],
                    length=40,
                ),
                maximum_notional_eur=None,
                reason_codes=(decision.reason_code,),
            )
            exit_pre_notification = (
                _notify_autonomous_event_safely(
                    settings,
                    "ORDER_SUBMITTING",
                    {
                        "intent_id": exit_intent.intent_id,
                        "market": market,
                        "side": "SELL",
                        "order_type": "MARKET",
                        "price": float(public_price),
                        "quantity": str(owned),
                        "notional_eur": float(
                            owned * public_price
                        ),
                        "strategy_id": PRIMARY_STRATEGY_ID,
                        "reason_code": decision.reason_code,
                    },
                )
            )
            try:
                exit_order = await client.submit_order(
                    exit_intent,
                    capability=preflight.capability,
                    estimated_price=public_price,
                    reconciled_owned_quantity=owned,
                    reconciled_total_exposure_eur=(
                        owned * public_price
                    ),
                    reconciled_open_positions=1,
                    exchange_minimum_order_eur=Decimal("5"),
                )
            except ReconciliationRequired:
                ambiguous_position = {
                    **reconciled_position,
                    "exit_client_order_id": (
                        client.client_order_id_for(
                            exit_intent.idempotency_key
                        )
                    ),
                    "exit_reason": decision.reason_code,
                    "exit_submitted_at": now,
                    "exit_status": "AMBIGUOUS_RECONCILIATION_REQUIRED",
                }
                atomic_write_json(
                    position_path,
                    {
                        "schema_version": "current_position_v1",
                        "status": "EXIT_PENDING_RECONCILIATION",
                        "position": ambiguous_position,
                        "orders_generated": 1,
                        "orders_submitted": 1,
                    },
                )
                notification = _notify_autonomous_event_safely(
                    settings,
                    "RECONCILIATION_MISMATCH",
                    {
                        "market": market,
                        "status": "EXIT_SUBMISSION_AMBIGUOUS",
                        "reason_code": (
                            "EXACT_CLIENT_ORDER_RECONCILIATION_REQUIRED"
                        ),
                    },
                )
                return {
                    **live,
                    "status": "RECONCILIATION_REQUIRED",
                    "cycle_status": "NO_DUPLICATE_ORDER",
                    "reason_code": "EXIT_SUBMISSION_AMBIGUOUS",
                    "notification": notification,
                    "private_exchange_requests": 4,
                    "orders_generated": 1,
                    "orders_submitted": 1,
                }
            except ExecutionBlocked:
                notification = _notify_autonomous_event_safely(
                    settings,
                    "ORDER_REJECTED",
                    {
                        "market": market,
                        "side": "SELL",
                        "order_type": "MARKET",
                        "strategy_id": PRIMARY_STRATEGY_ID,
                        "reason_code": "CANONICAL_EXECUTION_REJECTED",
                    },
                )
                return {
                    **live,
                    "status": "BLOCKED",
                    "cycle_status": "ORDER_REJECTED",
                    "reason_code": "CANONICAL_EXECUTION_REJECTED",
                    "notification": notification,
                    "private_exchange_requests": 4,
                    "orders_generated": 1,
                    "orders_submitted": 0,
                }
            exit_order_status = str(
                exit_order.get("status") or ""
            ).casefold()
            exit_post_notification = (
                _notify_autonomous_event_safely(
                    settings,
                    (
                        "ORDER_FILLED"
                        if exit_order_status == "filled"
                        else "ORDER_PARTIALLY_FILLED"
                        if exit_order_status
                        == "partiallyfilled"
                        else "ORDER_SUBMITTING"
                    ),
                    {
                        "order_id": exit_order.get("orderId"),
                        "market": market,
                        "side": "SELL",
                        "order_type": "MARKET",
                        "price": float(public_price),
                        "requested_quantity": str(owned),
                        "filled_quantity": str(
                            exit_order.get("filledAmount")
                            or "0"
                        ),
                        "remaining_quantity": exit_order.get("amountRemaining"),
                        "average_fill_price": exit_order.get("price")
                        or str(public_price),
                        "invested_eur": exit_order.get("filledAmountQuote"),
                        "fee": exit_order.get("feePaid"),
                        "strategy_id": PRIMARY_STRATEGY_ID,
                        "timeframe": PRIMARY_TIMEFRAME,
                        "venue_timestamp": exit_order.get("updated"),
                        "status": exit_order.get("status"),
                        "verification_source": "BITVAVO_REST_ORDER_RESPONSE",
                        "reason_code": decision.reason_code,
                    },
                )
            )
            closed_position = {
                **reconciled_position,
                "exit_order_id": exit_order.get("orderId"),
                "exit_client_order_id": exit_order.get(
                    "clientOrderId"
                ),
                "exit_reason": decision.reason_code,
                "exit_submitted_at": now,
            }
            atomic_write_json(
                position_path,
                {
                    "schema_version": "current_position_v1",
                    "status": "EXIT_PENDING_RECONCILIATION",
                    "position": closed_position,
                    "orders_generated": 1,
                    "orders_submitted": 1,
                },
            )
            notification_type = (
                "STOP_LOSS_HIT"
                if decision.reason_code == "STOP_LOSS_REACHED"
                else "TP2_REACHED"
                if decision.reason_code == "TP2_REACHED"
                else "EXIT"
            )
            notification = _notify_autonomous_event_safely(
                settings,
                notification_type,
                {
                    "market": market,
                    "status": decision.reason_code,
                    "price": float(public_price),
                    "quantity": str(owned),
                    "strategy_id": PRIMARY_STRATEGY_ID,
                    "order_id": exit_order.get("orderId"),
                },
            )
            return {
                **live,
                "status": "EXIT_SUBMITTED",
                "cycle_status": "ORDER_SUBMITTED",
                "reason_code": decision.reason_code,
                "order": {
                    "order_id": exit_order.get("orderId"),
                    "client_order_id": exit_order.get(
                        "clientOrderId"
                    ),
                    "market": market,
                    "status": exit_order.get("status"),
                },
                "notification": notification,
                "order_notifications": {
                    "pre_submit": exit_pre_notification,
                    "post_submit": exit_post_notification,
                },
                "private_exchange_requests": 4,
                "orders_generated": 1,
                "orders_submitted": 1,
            }
        snapshot = PortfolioSnapshot(
            equity_eur=float(live_equity),
            cash_eur=float(eur),
            day_start_equity_eur=float(day_start_equity),
            peak_equity_eur=float(peak_equity),
            trades_today=0,
            reconciled=True,
        )
        risk = RiskManager.from_settings(
            settings,
            kill_switch_path=(settings.paths.checkpoints_dir / "kill_switch.json"),
        ).assess_entry(
            market=market,
            entry_price=float(estimated_price),
            stop_price=float(opportunity["stop_loss"]),
            snapshot=snapshot,
            live_mode=True,
        )
        if not risk.approved:
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "NO_TRADE",
                "reason_code": "LIVE_BLOCKED_RISK_REJECTED",
                "risk_reason_codes": [
                    getattr(value, "value", str(value)) for value in risk.reason_codes
                ],
                "private_exchange_requests": 3,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        quantity = min(quantity, risk.approved_quantity)
        requested_notional = quantity * estimated_price
        canary_policy = CanaryPolicy.from_cap_limits(
            settings,
            maximum_order_eur=preflight.capability.maximum_order_eur,
            maximum_total_eur=preflight.capability.maximum_total_eur,
            maximum_open_positions=preflight.capability.maximum_open_positions,
            capital_level=capital_level,
            enabled=bool(live.get("operator_canary_authorized")),
        )
        canary = InstitutionalCanaryGuard(canary_policy).assess_buy(
            requested_notional_eur=requested_notional,
            current_total_exposure_eur=Decimal("0"),
            current_open_positions=0,
            exchange_minimum_order_eur=Decimal("5"),
        )
        if not canary.approved:
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "NO_TRADE",
                "reason_code": canary.reason_code,
                "private_exchange_requests": 3,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        quantity = canary.approved_notional_eur / estimated_price
        planned_risk_eur = quantity * abs(
            estimated_price - Decimal(str(opportunity["stop_loss"]))
        )
        if planned_risk_eur > Decimal(
            str(settings.execution.maximum_live_risk_per_trade_eur)
        ):
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "NO_TRADE",
                "reason_code": "LIVE_MAX_RISK_PER_TRADE_EUR_EXCEEDED",
                "planned_risk_eur": str(planned_risk_eur),
                "private_exchange_requests": 3,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        intent = OrderIntent(
            intent_id=stable_hash(
                [
                    "AUTONOMOUS_CANARY",
                    opportunity["opportunity_id"],
                    "BUY",
                ],
                length=32,
            ),
            idempotency_key=(f"autonomous-canary:{opportunity['opportunity_id']}:BUY"),
            market=market,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            strategy_id=PRIMARY_STRATEGY_ID,
            strategy_dna_hash=PRIMARY_STRATEGY_DNA,
            signal_id=str(opportunity["opportunity_id"]),
            portfolio_decision_id=stable_hash(
                [
                    "LIVE_ENTRY_DECISION",
                    PRIMARY_STRATEGY_DNA,
                    market,
                    opportunity["opportunity_id"],
                ],
                length=40,
            ),
            maximum_notional_eur=preflight.capability.maximum_order_eur,
            reason_codes=("HUMAN_APPROVED_LIVE_CANARY",),
        )
        canonical_costs = CanonicalCostModel.from_settings(settings)
        expected_net_edge = planned_target_net_edge(
            entry_price=estimated_price,
            target_price=Decimal(str(opportunity["take_profit_1"])),
            costs=canonical_costs,
        )
        raw_confidence = Decimal(str(opportunity.get("confidence") or "0.5"))
        if raw_confidence > Decimal("1"):
            raw_confidence /= Decimal("100")
        confidence = min(Decimal("1"), max(Decimal("0.01"), raw_confidence))
        try:
            canonical_plan = canonicalize_approved_buy_order(
                settings,
                intent,
                mark_price=estimated_price,
                current_quantity=owned,
                equity_eur=live_equity,
                approved_risk_eur=planned_risk_eur,
                expected_net_edge=expected_net_edge,
                confidence=confidence,
                family="RESIDUAL_REVERSAL",
                evidence_id=str(opportunity["opportunity_id"]),
                policy_version=(
                    f"autonomous_control_plane:{CONTROL_PLANE_VERSION}"
                ),
                account_state={
                    "equity_eur": str(live_equity),
                    "cash_eur": str(eur),
                    "day_start_equity_eur": str(day_start_equity),
                    "peak_equity_eur": str(peak_equity),
                    "reconciled": True,
                },
                portfolio_state={
                    "market": market,
                    "owned_quantity": str(owned),
                    "managed_position": False,
                    "maximum_total_exposure_eur": str(
                        preflight.capability.maximum_total_eur
                    ),
                },
                horizon_seconds=5 * 86_400,
            )
        except ExecutionBlocked:
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "NO_TRADE",
                "reason_code": "CANONICAL_BUY_CHAIN_REJECTED",
                "private_exchange_requests": 3,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        intent = canonical_plan.order
        pre_submit_notification = (
            _notify_autonomous_event_safely(
                settings,
                "ORDER_SUBMITTING",
                {
                    "intent_id": intent.intent_id,
                    "market": market,
                    "side": "BUY",
                    "order_type": "MARKET",
                    "price": float(estimated_price),
                    "quantity": str(quantity),
                    "notional_eur": float(
                        quantity * estimated_price
                    ),
                    "strategy_id": PRIMARY_STRATEGY_ID,
                },
            )
        )
        async def submit_reserved_rr_entry(
            fresh_portfolio: Mapping[str, Any],
        ) -> dict[str, Any]:
            return await client.submit_order(
                intent,
                capability=preflight.capability,
                estimated_price=estimated_price,
                reconciled_owned_quantity=owned,
                reconciled_total_exposure_eur=Decimal(
                    str(
                        fresh_portfolio[
                            "capacity_managed_exposure_eur"
                        ]
                    )
                ),
                reconciled_open_positions=int(
                    fresh_portfolio[
                        "capacity_managed_position_count"
                    ]
                ),
                exchange_minimum_order_eur=Decimal("5"),
                canonical_chain=canonical_plan.chain,
            )

        try:
            (
                reservation_approved,
                reservation_reason,
                reservation_portfolio,
                order,
            ) = await submit_level_2_buy_atomically(
                settings,
                requested_notional_eur=(quantity * estimated_price),
                submit_order=submit_reserved_rr_entry,
            )
            if not reservation_approved or order is None:
                return {
                    **live,
                    "status": "BLOCKED",
                    "cycle_status": "NO_TRADE",
                    "reason_code": reservation_reason,
                    "managed_portfolio": reservation_portfolio,
                    "private_exchange_requests": 3,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
        except ReconciliationRequired:
            ambiguous_position = {
                "strategy_id": PRIMARY_STRATEGY_ID,
                "strategy_dna_hash": PRIMARY_STRATEGY_DNA,
                "market": market,
                "side": "LONG",
                "entry_client_order_id": (
                    client.client_order_id_for(
                        intent.idempotency_key
                    )
                ),
                "entry_opportunity_id": opportunity[
                    "opportunity_id"
                ],
                "entry_price": float(estimated_price),
                "quantity": str(quantity),
                "stop_loss": opportunity["stop_loss"],
                "take_profit_1": opportunity["take_profit_1"],
                "take_profit_2": opportunity["take_profit_2"],
                "tp1_reached": False,
                "opened_at": now,
                "last_reconciled_at": None,
                "entry_status": (
                    "AMBIGUOUS_RECONCILIATION_REQUIRED"
                ),
            }
            atomic_write_json(
                position_path,
                {
                    "schema_version": "current_position_v1",
                    "status": "OPEN_PENDING_RECONCILIATION",
                    "position": ambiguous_position,
                    "orders_generated": 1,
                    "orders_submitted": 1,
                },
            )
            notification = _notify_autonomous_event_safely(
                settings,
                "RECONCILIATION_MISMATCH",
                {
                    "market": market,
                    "status": "ENTRY_SUBMISSION_AMBIGUOUS",
                    "reason_code": (
                        "EXACT_CLIENT_ORDER_RECONCILIATION_REQUIRED"
                    ),
                },
            )
            return {
                **live,
                "status": "RECONCILIATION_REQUIRED",
                "cycle_status": "NO_DUPLICATE_ORDER",
                "reason_code": "ENTRY_SUBMISSION_AMBIGUOUS",
                "notification": notification,
                "private_exchange_requests": 4,
                "orders_generated": 1,
                "orders_submitted": 1,
            }
        except ExecutionBlocked:
            notification = _notify_autonomous_event_safely(
                settings,
                "ORDER_REJECTED",
                {
                    "market": market,
                    "side": "BUY",
                    "order_type": "MARKET",
                    "strategy_id": PRIMARY_STRATEGY_ID,
                    "reason_code": "CANONICAL_EXECUTION_REJECTED",
                },
            )
            return {
                **live,
                "status": "BLOCKED",
                "cycle_status": "ORDER_REJECTED",
                "reason_code": "CANONICAL_EXECUTION_REJECTED",
                "notification": notification,
                "private_exchange_requests": 4,
                "orders_generated": 1,
                "orders_submitted": 0,
            }
        order_status = str(order.get("status") or "").casefold()
        post_submit_notification = _notify_autonomous_event_safely(
            settings,
            (
                "ORDER_FILLED"
                if order_status == "filled"
                else "ORDER_PARTIALLY_FILLED"
                if order_status == "partiallyfilled"
                else "ORDER_SUBMITTING"
            ),
            {
                "order_id": order.get("orderId"),
                "market": market,
                "side": "BUY",
                "order_type": "MARKET",
                "average_fill_price": float(
                    order.get("price") or estimated_price
                ),
                "requested_quantity": str(quantity),
                "filled_quantity": str(order.get("filledAmount") or "0"),
                "remaining_quantity": order.get("amountRemaining"),
                "invested_eur": order.get("filledAmountQuote"),
                "fee": order.get("feePaid"),
                "strategy_id": PRIMARY_STRATEGY_ID,
                "timeframe": PRIMARY_TIMEFRAME,
                "venue_timestamp": order.get("updated"),
                "status": order.get("status"),
                "verification_source": "BITVAVO_REST_ORDER_RESPONSE",
            },
        )
    position_payload = {
        "schema_version": "current_position_v1",
        "status": (
            "OPEN"
            if str(order.get("status") or "").casefold() == "filled"
            else "OPEN_PENDING_RECONCILIATION"
        ),
        "position": {
            "strategy_id": PRIMARY_STRATEGY_ID,
            "strategy_dna_hash": PRIMARY_STRATEGY_DNA,
            "market": market,
            "side": "LONG",
            "entry_order_id": order.get("orderId"),
            "entry_client_order_id": order.get("clientOrderId"),
            "entry_opportunity_id": opportunity["opportunity_id"],
            "entry_price": float(
                order.get("price") or estimated_price
            ),
            "quantity": str(
                order.get("filledAmount") or quantity
            ),
            "stop_loss": opportunity["stop_loss"],
            "take_profit_1": opportunity["take_profit_1"],
            "take_profit_2": opportunity["take_profit_2"],
            "tp1_reached": False,
            "opened_at": now,
            "last_reconciled_at": None,
        },
        "orders_generated": 1,
        "orders_submitted": 1,
    }
    atomic_write_json(
        settings.paths.output_dir
        / "reports"
        / "current_position.json",
        position_payload,
    )
    return {
        **live,
        "status": "SUBMITTED",
        "cycle_status": "ORDER_SUBMITTED",
        "reason_code": "AUTONOMOUS_CANARY_ORDER_SUBMITTED",
        "order": {
            "order_id": order.get("orderId"),
            "client_order_id": order.get("clientOrderId"),
            "market": order.get("market", market),
            "status": order.get("status"),
        },
        "notifications": {
            "pre_submit": pre_submit_notification,
            "post_submit": post_submit_notification,
        },
        "private_exchange_requests": 4,
        "orders_generated": 1,
        "orders_submitted": 1,
    }


__all__ = [
    "CONTROL_PLANE_VERSION",
    "LiveApprovalStatus",
    "LiveStrategyApproval",
    "LiveStrategyApprovalRegistry",
    "ManagedPositionDecision",
    "MarketRegime",
    "MarketRegimeClassifier",
    "Opportunity",
    "OpportunityScanner",
    "PRIMARY_STRATEGY_DNA",
    "PRIMARY_STRATEGY_ID",
    "RegimeSnapshot",
    "RoutedStrategy",
    "StrategyRegimeRouter",
    "build_autonomous_control_plane",
    "build_fresh_autonomous_control_plane",
    "bitvavo_entry_liquidity",
    "bitvavo_public_price",
    "decide_managed_position_action",
    "execute_autonomous_canary_once",
    "notify_autonomous_event_safely",
    "refresh_primary_daily_frames",
]
