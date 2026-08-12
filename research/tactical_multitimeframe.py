"""Causal tactical 15m/1h/2h strategy catalogue.

The module deliberately separates *strategy discovery* from live authority.
Every strategy uses closed execution candles, completed higher-timeframe
context and next-open execution in the canonical backtester.  Registering a
DNA here never grants order authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from research.strategies import Strategy, StrategyOutput, crossed_above, crossed_below
from utils.common import stable_hash

TACTICAL_ENGINE_VERSION = "1.1.0"


@dataclass(frozen=True, slots=True)
class TacticalStrategySpec:
    """Immutable economic identity for one tactical strategy."""

    strategy_id: str
    family: str
    timeframe: str
    confirmation_timeframe: str
    regime_timeframe: str
    mechanism: str
    context_policy: str
    stop_atr: float
    target_atr: float
    trailing_atr: float
    maximum_holding_bars: int
    expected_holding_period: str

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "engine_version": TACTICAL_ENGINE_VERSION,
                **asdict(self),
                "closed_candle_only": True,
                "next_open_execution": True,
                "long_only_spot": True,
            },
            length=64,
        )


def tactical_strategy_specs() -> tuple[TacticalStrategySpec, ...]:
    """Return independent preregistered 15m, 1h and 2h families."""

    rows = (
        # 15m: execution timing with fully closed 1h confirmation and 4h regime.
        TacticalStrategySpec(
            "TACTICAL_15M_TREND_PULLBACK",
            "TREND_PULLBACK",
            "15m",
            "1h",
            "4h",
            "trend_pullback",
            "TREND",
            1.75,
            3.5,
            2.0,
            96,
            "1h-24h",
        ),
        TacticalStrategySpec(
            "TACTICAL_15M_VWAP_RECLAIM",
            "VWAP_RECLAIM",
            "15m",
            "1h",
            "4h",
            "vwap_reclaim",
            "TREND_OR_NEUTRAL",
            1.5,
            3.0,
            1.75,
            64,
            "45m-16h",
        ),
        TacticalStrategySpec(
            "TACTICAL_15M_DONCHIAN_BREAKOUT",
            "DONCHIAN_BREAKOUT",
            "15m",
            "1h",
            "4h",
            "donchian_breakout",
            "TREND",
            2.0,
            4.5,
            2.25,
            160,
            "2h-2d",
        ),
        TacticalStrategySpec(
            "TACTICAL_15M_LIQUIDITY_SWEEP",
            "LIQUIDITY_SWEEP_RECOVERY",
            "15m",
            "1h",
            "4h",
            "liquidity_sweep",
            "RECOVERY",
            1.25,
            2.75,
            1.5,
            64,
            "45m-16h",
        ),
        TacticalStrategySpec(
            "TACTICAL_15M_VOLUME_EXPANSION",
            "VOLUME_EXPANSION",
            "15m",
            "1h",
            "4h",
            "volume_expansion",
            "TREND_OR_NEUTRAL",
            1.75,
            3.75,
            2.0,
            96,
            "1h-24h",
        ),
        TacticalStrategySpec(
            "TACTICAL_15M_FAILED_BREAKOUT_REVERSAL",
            "FAILED_BREAKOUT_REVERSAL",
            "15m",
            "1h",
            "4h",
            "failed_breakout",
            "RECOVERY",
            1.25,
            2.5,
            1.25,
            48,
            "30m-12h",
        ),
        TacticalStrategySpec(
            "TACTICAL_15M_RANGE_VWAP_REVERSION",
            "RANGE_VWAP_REVERSION",
            "15m",
            "1h",
            "4h",
            "range_vwap_reversion",
            "RANGE",
            1.5,
            2.75,
            0.0,
            48,
            "45m-12h",
        ),
        TacticalStrategySpec(
            "TACTICAL_15M_COMPRESSION_BREAKOUT",
            "VOLATILITY_COMPRESSION_BREAKOUT",
            "15m",
            "1h",
            "4h",
            "compression_breakout",
            "TREND_OR_NEUTRAL",
            1.75,
            4.0,
            2.0,
            96,
            "1h-24h",
        ),
        TacticalStrategySpec(
            "TACTICAL_15M_RS_LEADER_PULLBACK",
            "RS_LEADER_PULLBACK",
            "15m",
            "1h",
            "4h",
            "rs_leader_pullback",
            "TREND_OR_NEUTRAL",
            1.75,
            3.75,
            2.0,
            80,
            "1h-20h",
        ),
        TacticalStrategySpec(
            "TACTICAL_15M_BREAKOUT_RETEST",
            "BREAKOUT_RETEST",
            "15m",
            "1h",
            "4h",
            "breakout_retest",
            "TREND_OR_NEUTRAL",
            1.75,
            4.0,
            2.0,
            96,
            "1h-24h",
        ),
        # 1h: twelve distinct mechanisms.
        TacticalStrategySpec(
            "TACTICAL_1H_TREND_PULLBACK",
            "TREND_PULLBACK",
            "1h",
            "4h",
            "1d",
            "trend_pullback",
            "TREND",
            2.0,
            4.0,
            2.5,
            72,
            "6h-3d",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_DONCHIAN_BREAKOUT",
            "DONCHIAN_BREAKOUT",
            "1h",
            "4h",
            "1d",
            "donchian_breakout",
            "TREND",
            2.5,
            6.0,
            3.0,
            120,
            "12h-5d",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_COMPRESSION_BREAKOUT",
            "VOLATILITY_COMPRESSION_BREAKOUT",
            "1h",
            "4h",
            "1d",
            "compression_breakout",
            "TREND_OR_NEUTRAL",
            2.0,
            5.0,
            2.5,
            80,
            "8h-4d",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_FRACTAL_BREAKOUT",
            "CONFIRMED_FRACTAL_BREAKOUT",
            "1h",
            "4h",
            "1d",
            "fractal_breakout",
            "TREND",
            2.0,
            5.0,
            2.5,
            96,
            "8h-4d",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_VWAP_RECLAIM",
            "VWAP_RECLAIM",
            "1h",
            "4h",
            "1d",
            "vwap_reclaim",
            "TREND_OR_NEUTRAL",
            1.75,
            3.5,
            2.0,
            48,
            "4h-2d",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_RS_CONTINUATION",
            "RELATIVE_STRENGTH_CONTINUATION",
            "1h",
            "4h",
            "1d",
            "relative_strength",
            "TREND",
            2.0,
            4.5,
            2.5,
            72,
            "6h-3d",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_RANGE_REVERSION",
            "RANGE_MEAN_REVERSION",
            "1h",
            "4h",
            "1d",
            "range_reversion",
            "RANGE",
            1.5,
            2.5,
            0.0,
            30,
            "2h-24h",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_LIQUIDITY_SWEEP",
            "LIQUIDITY_SWEEP_RECOVERY",
            "1h",
            "4h",
            "1d",
            "liquidity_sweep",
            "RECOVERY",
            1.5,
            3.5,
            2.0,
            48,
            "3h-2d",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_POST_LIQUIDATION_RECOVERY",
            "POST_LIQUIDATION_RECOVERY",
            "1h",
            "4h",
            "1d",
            "post_liquidation_recovery",
            "RECOVERY",
            1.75,
            3.5,
            2.0,
            48,
            "3h-2d",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_MOMENTUM_ACCELERATION",
            "MOMENTUM_ACCELERATION",
            "1h",
            "4h",
            "1d",
            "momentum_acceleration",
            "TREND",
            2.0,
            4.5,
            2.5,
            60,
            "4h-3d",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_FAILED_BREAKOUT_REVERSAL",
            "FAILED_BREAKOUT_REVERSAL",
            "1h",
            "4h",
            "1d",
            "failed_breakout",
            "RECOVERY",
            1.5,
            3.0,
            1.5,
            36,
            "2h-36h",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_VOLUME_EXPANSION",
            "VOLUME_EXPANSION",
            "1h",
            "4h",
            "1d",
            "volume_expansion",
            "TREND_OR_NEUTRAL",
            2.0,
            4.0,
            2.0,
            60,
            "4h-3d",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_RS_LEADER_PULLBACK",
            "RS_LEADER_PULLBACK",
            "1h",
            "4h",
            "1d",
            "rs_leader_pullback",
            "TREND_OR_NEUTRAL",
            2.0,
            4.5,
            2.5,
            60,
            "4h-3d",
        ),
        TacticalStrategySpec(
            "TACTICAL_1H_BREAKOUT_RETEST",
            "BREAKOUT_RETEST",
            "1h",
            "4h",
            "1d",
            "breakout_retest",
            "TREND_OR_NEUTRAL",
            2.0,
            4.5,
            2.5,
            60,
            "4h-3d",
        ),
        # 2h: ten independent mechanisms.
        TacticalStrategySpec(
            "TACTICAL_2H_MTF_TREND_CONTINUATION",
            "MULTI_TIMEFRAME_TREND_CONTINUATION",
            "2h",
            "4h",
            "1d",
            "mtf_trend_continuation",
            "TREND",
            2.25,
            5.0,
            2.5,
            60,
            "12h-6d",
        ),
        TacticalStrategySpec(
            "TACTICAL_2H_DONCHIAN_ATR_FRACTAL",
            "DONCHIAN_ATR_FRACTAL",
            "2h",
            "4h",
            "1d",
            "donchian_atr_fractal",
            "TREND",
            4.0,
            10.0,
            3.0,
            120,
            "1d-10d",
        ),
        TacticalStrategySpec(
            "TACTICAL_2H_TREND_PULLBACK",
            "TREND_PULLBACK",
            "2h",
            "4h",
            "1d",
            "trend_pullback",
            "TREND",
            2.0,
            4.5,
            2.5,
            60,
            "12h-6d",
        ),
        TacticalStrategySpec(
            "TACTICAL_2H_BREAKOUT_RETEST",
            "BREAKOUT_RETEST",
            "2h",
            "4h",
            "1d",
            "breakout_retest",
            "TREND_OR_NEUTRAL",
            2.0,
            4.5,
            2.5,
            60,
            "12h-6d",
        ),
        TacticalStrategySpec(
            "TACTICAL_2H_VOLATILITY_EXPANSION",
            "VOLATILITY_EXPANSION",
            "2h",
            "4h",
            "1d",
            "volatility_expansion",
            "TREND_OR_NEUTRAL",
            2.25,
            5.0,
            2.5,
            48,
            "8h-4d",
        ),
        TacticalStrategySpec(
            "TACTICAL_2H_RS_ROTATION",
            "RELATIVE_STRENGTH_ROTATION",
            "2h",
            "4h",
            "1d",
            "relative_strength",
            "TREND",
            2.0,
            4.5,
            2.5,
            60,
            "12h-6d",
        ),
        TacticalStrategySpec(
            "TACTICAL_2H_STRUCTURE_CONTINUATION",
            "MARKET_STRUCTURE_CONTINUATION",
            "2h",
            "4h",
            "1d",
            "structure_continuation",
            "TREND",
            2.25,
            5.5,
            2.5,
            72,
            "12h-8d",
        ),
        TacticalStrategySpec(
            "TACTICAL_2H_RANGE_BREAKOUT",
            "RANGE_BREAKOUT",
            "2h",
            "4h",
            "1d",
            "range_breakout",
            "TREND_OR_NEUTRAL",
            2.0,
            5.0,
            2.5,
            60,
            "12h-6d",
        ),
        TacticalStrategySpec(
            "TACTICAL_2H_DEFENSIVE_RECOVERY",
            "DEFENSIVE_BTC_ETH_RECOVERY",
            "2h",
            "4h",
            "1d",
            "defensive_recovery",
            "RECOVERY",
            1.75,
            3.5,
            2.0,
            48,
            "8h-4d",
        ),
        TacticalStrategySpec(
            "TACTICAL_2H_CROSS_SECTIONAL_MOMENTUM",
            "CROSS_SECTIONAL_MOMENTUM",
            "2h",
            "4h",
            "1d",
            "cross_sectional_momentum",
            "TREND",
            2.0,
            4.5,
            2.5,
            60,
            "12h-6d",
        ),
    )
    identities = {row.dna_hash for row in rows}
    if len(rows) != 34 or len(identities) != len(rows):
        raise RuntimeError("tactical strategy catalogue identity drift")
    return rows


def _boolean(
    frame: pd.DataFrame,
    name: str,
    *,
    default: bool = False,
) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[name].astype("boolean").fillna(default).astype(bool)


def _numeric(
    frame: pd.DataFrame,
    name: str,
    *,
    default: float = 0.0,
) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default)


class TacticalMultiTimeframeStrategy(Strategy):
    """One immutable tactical mechanism with explicit timeframe alignment."""

    parameter_space: dict[str, tuple[Any, ...]] = {}
    uses_intelligence = False

    def __init__(self, spec: TacticalStrategySpec) -> None:
        self.spec = spec
        self.strategy_id = spec.strategy_id
        self.family = spec.family
        self.description = (
            f"{spec.timeframe} {spec.family} with {spec.confirmation_timeframe} "
            f"confirmation and {spec.regime_timeframe} regime context."
        )
        self.defaults = {
            "stop_atr": spec.stop_atr,
            "target_atr": spec.target_atr,
            "trailing_atr": spec.trailing_atr,
            "maximum_holding_bars": spec.maximum_holding_bars,
        }

    def _context(
        self,
        features: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        confirmation = _boolean(
            features,
            f"htf_{self.spec.confirmation_timeframe}_trend_bullish",
        )
        regime = _boolean(
            features,
            f"htf_{self.spec.regime_timeframe}_trend_bullish",
        )
        conflicts = (~confirmation).astype(int) + (~regime).astype(int)
        if self.spec.context_policy == "TREND":
            allowed = confirmation & regime
        elif self.spec.context_policy == "TREND_OR_NEUTRAL":
            allowed = confirmation | regime
        elif self.spec.context_policy == "RANGE":
            allowed = ~_boolean(features, "bull_regime") & ~_boolean(
                features,
                "bear_regime",
            )
        elif self.spec.context_policy == "RECOVERY":
            allowed = confirmation | ~_boolean(features, "bear_regime")
        else:  # pragma: no cover - catalogue invariant
            raise ValueError(f"unknown context policy: {self.spec.context_policy}")
        alignment = (2.0 - conflicts.astype(float)) / 2.0
        return allowed, alignment.clip(0.0, 1.0), conflicts

    def _mechanism(
        self,
        features: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series, str, str]:
        close = _numeric(features, "close")
        ema20 = _numeric(features, "ema_20")
        ema50 = _numeric(features, "ema_50")
        ema200 = _numeric(features, "ema_200")
        rsi = _numeric(features, "rsi_14", default=50.0)
        volume_z = _numeric(features, "volume_zscore_20")
        relative_volume = _numeric(features, "relative_volume_20")
        mechanism = self.spec.mechanism
        if mechanism in {"trend_pullback", "mtf_trend_continuation"}:
            entry = (
                (ema20 > ema50)
                & (ema50 > ema200)
                & crossed_above(close, ema20)
                & rsi.between(40.0, 70.0)
            )
            exit_ = crossed_below(close, ema50)
        elif mechanism in {"donchian_breakout", "donchian_atr_fractal"}:
            lookback = 55 if mechanism == "donchian_breakout" else 120
            exit_lookback = 20 if mechanism == "donchian_breakout" else 36
            high = _numeric(
                features,
                f"donchian_high_{lookback}",
                default=float("nan"),
            )
            if high.isna().all():
                high = _numeric(features, "high").shift(1).rolling(lookback).max()
            low = _numeric(features, "low").shift(1).rolling(exit_lookback).min()
            entry = crossed_above(close, high) & (volume_z >= 0.0)
            if mechanism == "donchian_atr_fractal":
                entry &= _numeric(
                    features,
                    "multi_timeframe_fractal_alignment",
                ) >= 0.0
            exit_ = crossed_below(close, low)
        elif mechanism == "compression_breakout":
            compressed = _boolean(
                features,
                "bollinger_keltner_squeeze_state",
            ).shift(1, fill_value=False)
            entry = (
                compressed
                & crossed_above(close, _numeric(features, "donchian_high_20"))
                & (relative_volume >= 1.0)
            )
            exit_ = crossed_below(close, ema20)
        elif mechanism == "fractal_breakout":
            level = _numeric(
                features,
                "confirmed_fractal_high_price",
                default=float("nan"),
            ).ffill()
            entry = crossed_above(close, level) & (volume_z >= 0.0)
            exit_ = _boolean(features, "bearish_choch") | crossed_below(
                close,
                ema20,
            )
        elif mechanism == "vwap_reclaim":
            entry = (
                _boolean(features, "vwap_reclaim")
                & (relative_volume >= 0.9)
                & (rsi >= 40.0)
            )
            exit_ = crossed_below(close, _numeric(features, "vwap_20"))
        elif mechanism in {"relative_strength", "cross_sectional_momentum"}:
            relative = _numeric(features, "btc_relative_momentum_20")
            threshold = 0.0 if mechanism == "relative_strength" else 0.01
            entry = crossed_above(relative, threshold) & (close > ema50)
            exit_ = crossed_below(relative, 0.0) | crossed_below(close, ema50)
        elif mechanism == "range_reversion":
            entry = (
                (close < _numeric(features, "bollinger_lower"))
                & crossed_above(rsi, 28.0)
            )
            exit_ = crossed_above(close, _numeric(features, "bollinger_middle"))
        elif mechanism == "range_vwap_reversion":
            atr = _numeric(features, "atr_14").replace(0.0, float("nan"))
            vwap = _numeric(features, "vwap_20")
            efficiency = (close - close.shift(20)).abs() / (
                close.diff().abs().rolling(20).sum().replace(0.0, float("nan"))
            )
            vwap_distance_atr = (close - vwap) / atr
            entry = (
                (efficiency < 0.25)
                & (vwap_distance_atr < -0.45)
                & crossed_above(rsi, 32.0)
                & (relative_volume >= 0.75)
            )
            exit_ = crossed_above(close, vwap) | crossed_above(rsi, 58.0)
        elif mechanism == "rs_leader_pullback":
            relative = _numeric(features, "btc_relative_momentum_20")
            leader_floor = relative.rolling(48).quantile(0.70).shift(1)
            entry = (
                (relative > 0.0)
                & (relative >= leader_floor)
                & (close > ema50)
                & crossed_above(close, ema20)
                & rsi.between(40.0, 70.0)
            )
            exit_ = crossed_below(relative, 0.0) | crossed_below(close, ema50)
        elif mechanism == "liquidity_sweep":
            entry = (
                _boolean(features, "bullish_liquidity_sweep")
                & (rsi <= 50.0)
                & (relative_volume >= 0.8)
            )
            exit_ = _boolean(features, "bearish_liquidity_sweep") | _boolean(
                features,
                "bearish_choch",
            )
        elif mechanism == "post_liquidation_recovery":
            prior_shock = (
                _numeric(features, "return_1").shift(1) < -0.03
            ) & (volume_z.shift(1) > 1.5)
            entry = prior_shock & (close > _numeric(features, "high").shift(1)) & (
                rsi > rsi.shift(1)
            )
            exit_ = crossed_below(close, ema20)
        elif mechanism == "momentum_acceleration":
            acceleration = _numeric(features, "momentum_acceleration")
            entry = crossed_above(acceleration, 0.0) & (
                _numeric(features, "roc_12") > 0.0
            ) & (relative_volume >= 1.0)
            exit_ = crossed_below(acceleration, 0.0) | crossed_below(close, ema20)
        elif mechanism == "failed_breakout":
            entry = _boolean(features, "failed_breakout_reclaim") & (
                rsi <= 55.0
            )
            exit_ = _boolean(features, "bearish_liquidity_sweep") | crossed_below(
                close,
                ema20,
            )
        elif mechanism in {"volume_expansion", "volatility_expansion"}:
            expansion = (
                _numeric(features, "true_range")
                > 1.35 * _numeric(features, "atr_14").shift(1)
            )
            entry = (
                expansion
                & (close > _numeric(features, "high").shift(1))
                & (relative_volume >= 1.25)
            )
            exit_ = crossed_below(close, ema20)
        elif mechanism == "breakout_retest":
            level = _numeric(features, "donchian_high_20")
            prior_break = close.shift(1) > level.shift(1)
            entry = prior_break & (close >= level * 0.995) & (close > ema20)
            exit_ = crossed_below(close, ema50)
        elif mechanism == "structure_continuation":
            entry = (
                (_boolean(features, "bullish_bos") | _boolean(features, "bullish_choch"))
                & (_numeric(features, "swing_trend") > 0)
                & (volume_z >= 0.0)
            )
            exit_ = _boolean(features, "bearish_bos") | _boolean(
                features,
                "bearish_choch",
            )
        elif mechanism == "range_breakout":
            range_high = _numeric(features, "high").shift(1).rolling(24).max()
            range_low = _numeric(features, "low").shift(1).rolling(12).min()
            entry = crossed_above(close, range_high) & (relative_volume >= 1.1)
            exit_ = crossed_below(close, range_low)
        elif mechanism == "defensive_recovery":
            entry = (
                crossed_above(rsi, 35.0)
                & crossed_above(close, ema20)
                & (relative_volume >= 0.8)
            )
            exit_ = crossed_below(close, ema50) | crossed_above(rsi, 65.0)
        else:  # pragma: no cover - catalogue invariant
            raise ValueError(f"unsupported tactical mechanism: {mechanism}")
        return entry, exit_, f"{self.spec.family}_ENTRY", f"{self.spec.family}_EXIT"

    def generate(
        self,
        features: pd.DataFrame,
        parameters: dict[str, Any] | None = None,
    ) -> StrategyOutput:
        selected = self.parameters(parameters)
        mechanism_entry, exit_, entry_reason, exit_reason = self._mechanism(
            features,
        )
        context_preferred, alignment, conflicts = self._context(features)
        # Higher timeframes are context and sizing inputs, never standalone
        # permission.  The active-trading control layer applies the complete
        # 15m/1h/2h/4h/1d/1W weighted score and routes a valid mechanism to a
        # trend or countertrend policy before execution economics are checked.
        avoid = pd.Series(False, index=features.index, dtype=bool)
        size = (0.60 + 0.40 * alignment).clip(0.40, 1.0)
        return self._output(
            features,
            entry=mechanism_entry,
            exit=exit_,
            avoid=avoid,
            size_multiplier=size,
            parameters=selected,
            entry_reason=entry_reason,
            exit_reason=exit_reason,
            metadata={
                "strategy_dna_hash": self.spec.dna_hash,
                "entry_timeframe": self.spec.timeframe,
                "confirmation_timeframe": self.spec.confirmation_timeframe,
                "regime_timeframe": self.spec.regime_timeframe,
                "timeframe_alignment_score": alignment,
                "timeframe_conflict_count": conflicts,
                "legacy_context_preferred": context_preferred,
                "higher_timeframe_context_is_soft": True,
                "standalone_1d_1w_veto": False,
                "closed_candle_only": True,
                "next_open_execution": True,
            },
        )


def tactical_strategy_registry() -> dict[str, TacticalMultiTimeframeStrategy]:
    return {
        spec.strategy_id: TacticalMultiTimeframeStrategy(spec)
        for spec in tactical_strategy_specs()
    }


def tactical_catalogue_payload() -> dict[str, Any]:
    rows = [
        {
            **asdict(spec),
            "strategy_dna_hash": spec.dna_hash,
            "status": "SHADOW_PENDING_EXACT_VALIDATION",
            "live_authority_granted": False,
            "integrity": {
                "closed_candle_only": True,
                "higher_timeframe_backward_asof": True,
                "next_open_execution": True,
                "lookahead": False,
                "repainting": False,
                "long_only_spot": True,
                "bounded_stop": True,
                "valid_exit": True,
            },
        }
        for spec in tactical_strategy_specs()
    ]
    return {
        "schema_version": "tactical_multitimeframe_catalogue_v1",
        "engine_version": TACTICAL_ENGINE_VERSION,
        "strategy_count": len(rows),
        "strategy_count_15m": sum(row["timeframe"] == "15m" for row in rows),
        "strategy_count_1h": sum(row["timeframe"] == "1h" for row in rows),
        "strategy_count_2h": sum(row["timeframe"] == "2h" for row in rows),
        "independent_families_15m": len(
            {row["family"] for row in rows if row["timeframe"] == "15m"}
        ),
        "independent_families_1h": len(
            {row["family"] for row in rows if row["timeframe"] == "1h"}
        ),
        "independent_families_2h": len(
            {row["family"] for row in rows if row["timeframe"] == "2h"}
        ),
        "strategies": rows,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


__all__ = [
    "TACTICAL_ENGINE_VERSION",
    "TacticalMultiTimeframeStrategy",
    "TacticalStrategySpec",
    "tactical_catalogue_payload",
    "tactical_strategy_registry",
    "tactical_strategy_specs",
]
