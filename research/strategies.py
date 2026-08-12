"""Registered long-only crypto spot strategy families and risk overlays."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

from core.contracts import StrategyMetadata


def crossed_above(left: pd.Series, right: pd.Series | float) -> pd.Series:
    right_series = (
        right
        if isinstance(right, pd.Series)
        else pd.Series(float(right), index=left.index)
    )
    return (left > right_series) & (left.shift(1) <= right_series.shift(1))


def crossed_below(left: pd.Series, right: pd.Series | float) -> pd.Series:
    right_series = (
        right
        if isinstance(right, pd.Series)
        else pd.Series(float(right), index=left.index)
    )
    return (left < right_series) & (left.shift(1) >= right_series.shift(1))


@dataclass(frozen=True)
class StrategyOutput:
    entry: pd.Series
    exit: pd.Series
    avoid: pd.Series
    reduce: pd.Series
    stop_distance: pd.Series
    target_distance: pd.Series
    trailing_distance: pd.Series
    size_multiplier: pd.Series
    maximum_holding_bars: int | None
    entry_reason: str
    exit_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, index: pd.Index) -> "StrategyOutput":
        fields = (
            self.entry,
            self.exit,
            self.avoid,
            self.reduce,
            self.stop_distance,
            self.target_distance,
            self.trailing_distance,
            self.size_multiplier,
        )
        if any(not value.index.equals(index) for value in fields):
            raise ValueError("strategy output index does not match feature index")
        if (self.stop_distance.dropna() <= 0).any():
            raise ValueError("stop distances must be positive")
        if (self.target_distance.dropna() <= 0).any():
            raise ValueError("target distances must be positive")
        if (self.trailing_distance.dropna() < 0).any():
            raise ValueError("trailing distances cannot be negative")
        if ((self.size_multiplier < 0) | (self.size_multiplier > 1)).any():
            raise ValueError("size multipliers must be between zero and one")
        return self


class Strategy(ABC):
    strategy_id: ClassVar[str]
    family: ClassVar[str]
    description: ClassVar[str]
    defaults: ClassVar[dict[str, Any]]
    parameter_space: ClassVar[dict[str, tuple[Any, ...]]]
    uses_intelligence: ClassVar[bool] = False

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_id=self.strategy_id,
            family=self.family,
            description=self.description,
            parameter_space=self.parameter_space,
            uses_intelligence=self.uses_intelligence,
        )

    def parameters(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        selected = dict(self.defaults)
        overrides = overrides or {}
        unknown = set(overrides) - set(selected)
        if unknown:
            raise ValueError(f"unknown parameters for {self.strategy_id}: {sorted(unknown)}")
        selected.update(overrides)
        self.validate_parameters(selected)
        return selected

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        if parameters["stop_atr"] <= 0 or parameters["target_atr"] <= 0:
            raise ValueError("stop_atr and target_atr must be positive")
        if parameters.get("trailing_atr", 0) < 0:
            raise ValueError("trailing_atr cannot be negative")

    def _output(
        self,
        features: pd.DataFrame,
        *,
        entry: pd.Series,
        exit: pd.Series,
        parameters: dict[str, Any],
        avoid: pd.Series | None = None,
        reduce: pd.Series | None = None,
        size_multiplier: pd.Series | None = None,
        entry_reason: str,
        exit_reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyOutput:
        index = features.index
        safe_entry = entry.reindex(index).fillna(False).astype(bool)
        safe_exit = exit.reindex(index).fillna(False).astype(bool)
        safe_avoid = (
            avoid.reindex(index).fillna(False).astype(bool)
            if avoid is not None
            else pd.Series(False, index=index)
        )
        safe_reduce = (
            reduce.reindex(index).fillna(False).astype(bool)
            if reduce is not None
            else pd.Series(False, index=index)
        )
        safe_entry &= ~safe_exit & ~safe_avoid
        atr_value = features["atr_14"].astype(float)
        stop = atr_value * float(parameters["stop_atr"])
        target = atr_value * float(parameters["target_atr"])
        trailing_multiple = float(parameters.get("trailing_atr", 0.0))
        trailing = atr_value * trailing_multiple
        sizes = (
            size_multiplier.reindex(index).fillna(0.0).clip(0.0, 1.0)
            if size_multiplier is not None
            else pd.Series(1.0, index=index)
        )
        sizes = sizes.where(~safe_avoid, 0.0)
        return StrategyOutput(
            entry=safe_entry,
            exit=safe_exit,
            avoid=safe_avoid,
            reduce=safe_reduce,
            stop_distance=stop,
            target_distance=target,
            trailing_distance=trailing,
            size_multiplier=sizes,
            maximum_holding_bars=parameters.get("maximum_holding_bars"),
            entry_reason=entry_reason,
            exit_reason=exit_reason,
            metadata=metadata or {},
        ).validate(index)

    @abstractmethod
    def generate(
        self,
        features: pd.DataFrame,
        parameters: dict[str, Any] | None = None,
    ) -> StrategyOutput: ...


class EmaTrendPullback(Strategy):
    strategy_id = "ema_trend_pullback"
    family = "trend_pullback"
    description = "Enter a resumed EMA uptrend after a controlled pullback."
    defaults = {
        "fast_period": 20,
        "slow_period": 50,
        "rsi_floor": 40.0,
        "stop_atr": 2.0,
        "target_atr": 4.0,
        "trailing_atr": 2.5,
        "maximum_holding_bars": 80,
    }
    parameter_space = {
        "rsi_floor": (35.0, 40.0, 45.0),
        "stop_atr": (1.5, 2.0, 2.5),
        "target_atr": (3.0, 4.0, 5.0),
    }

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        super().validate_parameters(parameters)
        if parameters["fast_period"] >= parameters["slow_period"]:
            raise ValueError("fast EMA period must be below slow EMA period")

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        trend = (
            (features["ema_20"] > features["ema_50"])
            & (features["ema_50"] > features["ema_200"])
            & (features["ema_50_slope"] > 0)
        )
        entry = trend & crossed_above(features["close"], features["ema_20"]) & (
            features["rsi_14"] >= p["rsi_floor"]
        )
        exit = crossed_below(features["close"], features["ema_50"]) | features["bear_regime"]
        return self._output(
            features,
            entry=entry,
            exit=exit,
            parameters=p,
            entry_reason="EMA_PULLBACK_RESUMPTION",
            exit_reason="EMA_TREND_INVALIDATED",
        )


class EmaCrossoverTrendFilter(Strategy):
    strategy_id = "ema_crossover_trend_filter"
    family = "trend"
    description = "EMA crossover only when the long-term crypto trend is positive."
    defaults = {
        "fast_period": 20,
        "slow_period": 50,
        "stop_atr": 2.5,
        "target_atr": 5.0,
        "trailing_atr": 3.0,
        "maximum_holding_bars": 120,
    }
    parameter_space = {
        "fast_period": (10, 20, 30),
        "slow_period": (40, 50, 80),
        "stop_atr": (2.0, 2.5, 3.0),
    }

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        super().validate_parameters(parameters)
        if parameters["fast_period"] >= parameters["slow_period"]:
            raise ValueError("fast EMA period must be below slow EMA period")

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        fast = features.get(f"ema_{p['fast_period']}", features["close"].ewm(span=p["fast_period"], adjust=False).mean())
        slow = features.get(f"ema_{p['slow_period']}", features["close"].ewm(span=p["slow_period"], adjust=False).mean())
        entry = crossed_above(fast, slow) & (features["close"] > features["ema_200"])
        exit = crossed_below(fast, slow) | (features["close"] < features["ema_200"])
        return self._output(features, entry=entry, exit=exit, parameters=p, entry_reason="EMA_BULLISH_CROSS", exit_reason="EMA_BEARISH_CROSS")


class DonchianBreakout(Strategy):
    strategy_id = "donchian_breakout"
    family = "breakout"
    description = "Causal breakout above the prior Donchian high."
    defaults = {
        "period": 55,
        "exit_period": 20,
        "stop_atr": 2.5,
        "target_atr": 6.0,
        "trailing_atr": 3.0,
        "maximum_holding_bars": 160,
    }
    parameter_space = {
        "period": (20, 55),
        "stop_atr": (2.0, 2.5, 3.0),
        "target_atr": (4.0, 6.0, 8.0),
    }

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        high = features.get(f"donchian_high_{p['period']}", features["high"].rolling(p["period"]).max().shift(1))
        low = features.get(f"donchian_low_{p['exit_period']}", features["low"].rolling(p["exit_period"]).min().shift(1))
        entry = crossed_above(features["close"], high) & (features["adx_14"] >= 18)
        exit = crossed_below(features["close"], low)
        return self._output(features, entry=entry, exit=exit, parameters=p, entry_reason="DONCHIAN_BREAKOUT", exit_reason="DONCHIAN_EXIT")


class MultiTimeframeDonchianVolumeBreakout(Strategy):
    """4h breakout gated by the last fully closed 1d short-trend state."""

    strategy_id = "mtf_1d_4h_donchian_rvol"
    family = "multi_timeframe_breakout"
    description = (
        "Enter a 4h Donchian breakout with relative-volume confirmation only "
        "when the last fully closed 1d candle has a rising EMA50 trend."
    )
    required_higher_timeframes: ClassVar[tuple[str, ...]] = ("1d",)
    defaults = {
        "period": 20,
        "exit_period": 10,
        "relative_volume_threshold": 1.1,
        "stop_atr": 2.0,
        "target_atr": 5.0,
        "trailing_atr": 2.5,
        "maximum_holding_bars": 120,
    }
    parameter_space = {
        "period": (20, 55),
        "relative_volume_threshold": (1.0, 1.1, 1.25),
        "stop_atr": (1.5, 2.0, 2.5),
        "target_atr": (4.0, 5.0, 6.0),
    }

    def generate(
        self,
        features: pd.DataFrame,
        parameters: dict[str, Any] | None = None,
    ) -> StrategyOutput:
        p = self.parameters(parameters)
        high = features.get(
            f"donchian_high_{p['period']}",
            features["high"].rolling(p["period"]).max().shift(1),
        )
        low = features.get(
            f"donchian_low_{p['exit_period']}",
            features["low"].rolling(p["exit_period"]).min().shift(1),
        )
        context = features.get("htf_1d_trend_bullish")
        if context is None:
            context = pd.Series(False, index=features.index)
        entry = (
            crossed_above(features["close"], high)
            & context.fillna(False).astype(bool)
            & (
                features["relative_volume_20"]
                >= float(p["relative_volume_threshold"])
            )
        )
        exit = crossed_below(features["close"], low) | ~context.fillna(False)
        return self._output(
            features,
            entry=entry,
            exit=exit,
            parameters=p,
            entry_reason="MTF_1D_TREND_4H_DONCHIAN_RVOL",
            exit_reason="MTF_DONCHIAN_OR_DAILY_TREND_EXIT",
            metadata={
                "execution_timeframe": "4h",
                "context_timeframes": ["1d"],
                "alignment": "BACKWARD_ASOF_AFTER_SOURCE_CANDLE_CLOSE",
            },
        )


class BollingerMeanReversion(Strategy):
    strategy_id = "bollinger_mean_reversion"
    family = "mean_reversion"
    description = "Buy an oversold lower-band excursion in a non-bearish regime."
    defaults = {
        "rsi_threshold": 30.0,
        "stop_atr": 1.5,
        "target_atr": 2.5,
        "trailing_atr": 0.0,
        "maximum_holding_bars": 30,
    }
    parameter_space = {
        "rsi_threshold": (25.0, 30.0, 35.0),
        "stop_atr": (1.0, 1.5, 2.0),
        "target_atr": (2.0, 2.5, 3.0),
    }

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        oversold = (features["close"] < features["bollinger_lower"]) & (
            features["rsi_14"] < p["rsi_threshold"]
        )
        entry = oversold & ~features["bear_regime"]
        exit = crossed_above(features["close"], features["bollinger_middle"]) | features["bear_regime"]
        return self._output(features, entry=entry, exit=exit, parameters=p, entry_reason="BOLLINGER_OVERSOLD", exit_reason="BOLLINGER_MEAN_REACHED")


class RsiMeanReversion(Strategy):
    strategy_id = "rsi_mean_reversion"
    family = "mean_reversion"
    description = "Enter when RSI recovers from oversold above a long-term trend filter."
    defaults = {
        "entry_rsi": 30.0,
        "exit_rsi": 58.0,
        "stop_atr": 1.75,
        "target_atr": 3.0,
        "trailing_atr": 0.0,
        "maximum_holding_bars": 40,
    }
    parameter_space = {
        "entry_rsi": (25.0, 30.0, 35.0),
        "exit_rsi": (50.0, 58.0, 65.0),
        "stop_atr": (1.5, 1.75, 2.0),
    }

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        entry = crossed_above(features["rsi_14"], p["entry_rsi"]) & (
            features["close"] > features["ema_200"]
        )
        exit = crossed_above(features["rsi_14"], p["exit_rsi"]) | (
            features["close"] < features["ema_50"]
        )
        return self._output(features, entry=entry, exit=exit, parameters=p, entry_reason="RSI_OVERSOLD_RECOVERY", exit_reason="RSI_MEAN_REVERSION_COMPLETE")


class ConnorsRsiPullback(Strategy):
    strategy_id = "connors_rsi_pullback"
    family = "mean_reversion"
    description = "Short-horizon pullback entry using Connors RSI in an uptrend."
    defaults = {
        "entry_threshold": 20.0,
        "exit_threshold": 70.0,
        "stop_atr": 1.5,
        "target_atr": 2.5,
        "trailing_atr": 0.0,
        "maximum_holding_bars": 20,
    }
    parameter_space = {
        "entry_threshold": (10.0, 20.0, 30.0),
        "exit_threshold": (60.0, 70.0, 80.0),
    }

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        entry = crossed_above(features["connors_rsi"], p["entry_threshold"]) & (
            features["close"] > features["ema_200"]
        )
        exit = crossed_above(features["connors_rsi"], p["exit_threshold"]) | features["bear_regime"]
        return self._output(features, entry=entry, exit=exit, parameters=p, entry_reason="CONNORS_RSI_PULLBACK", exit_reason="CONNORS_RSI_RECOVERED")


class VolatilitySqueezeBreakout(Strategy):
    strategy_id = "volatility_squeeze_breakout"
    family = "breakout"
    description = "Enter an upside breakout following Bollinger/Keltner compression."
    defaults = {
        "width_quantile": 0.25,
        "stop_atr": 2.0,
        "target_atr": 4.5,
        "trailing_atr": 2.5,
        "maximum_holding_bars": 80,
    }
    parameter_space = {
        "width_quantile": (0.15, 0.25, 0.35),
        "stop_atr": (1.5, 2.0, 2.5),
    }

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        threshold = features["bollinger_width"].rolling(100).quantile(p["width_quantile"])
        squeeze = (
            (features["bollinger_width"] <= threshold)
            & (features["bollinger_upper"] < features["keltner_upper"])
            & (features["bollinger_lower"] > features["keltner_lower"])
        )
        entry = squeeze.shift(1, fill_value=False) & crossed_above(
            features["close"], features["donchian_high_20"]
        )
        exit = crossed_below(features["close"], features["ema_20"])
        return self._output(features, entry=entry, exit=exit, parameters=p, entry_reason="VOLATILITY_SQUEEZE_BREAKOUT", exit_reason="SQUEEZE_BREAKOUT_FAILED")


class AtrExpansionBreakout(Strategy):
    strategy_id = "atr_expansion_breakout"
    family = "breakout"
    description = "Enter an upside range break during ATR expansion."
    defaults = {
        "expansion_multiple": 1.4,
        "stop_atr": 2.0,
        "target_atr": 4.0,
        "trailing_atr": 2.5,
        "maximum_holding_bars": 60,
    }
    parameter_space = {
        "expansion_multiple": (1.2, 1.4, 1.6),
        "stop_atr": (1.5, 2.0, 2.5),
    }

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        entry = (
            (features["true_range"] > p["expansion_multiple"] * features["atr_14"].shift(1))
            & (features["close"] > features["high"].shift(1))
            & (features["volume_zscore_20"] > 0)
        )
        exit = crossed_below(features["close"], features["ema_20"])
        return self._output(features, entry=entry, exit=exit, parameters=p, entry_reason="ATR_EXPANSION_BREAKOUT", exit_reason="ATR_BREAKOUT_TREND_LOST")


class LiquiditySweepReversal(Strategy):
    strategy_id = "liquidity_sweep_reversal"
    family = "market_structure"
    description = "Enter after a confirmed bullish liquidity sweep and reclaim."
    defaults = {
        "maximum_rsi": 48.0,
        "stop_atr": 1.5,
        "target_atr": 3.5,
        "trailing_atr": 2.0,
        "maximum_holding_bars": 50,
    }
    parameter_space = {
        "maximum_rsi": (40.0, 48.0, 55.0),
        "stop_atr": (1.25, 1.5, 2.0),
    }

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        entry = (
            features["bullish_liquidity_sweep"]
            & features["discount_zone"]
            & (features["rsi_14"] <= p["maximum_rsi"])
        )
        exit = features["bearish_liquidity_sweep"] | features["bearish_choch"]
        return self._output(features, entry=entry, exit=exit, parameters=p, entry_reason="BULLISH_LIQUIDITY_SWEEP", exit_reason="BEARISH_STRUCTURE_WARNING")


class BosChochContinuation(Strategy):
    strategy_id = "bos_choch_continuation"
    family = "market_structure"
    description = "Continue a confirmed bullish break or change of character."
    defaults = {
        "stop_atr": 2.0,
        "target_atr": 5.0,
        "trailing_atr": 2.5,
        "maximum_holding_bars": 100,
    }
    parameter_space = {
        "stop_atr": (1.5, 2.0, 2.5),
        "target_atr": (4.0, 5.0, 6.0),
    }

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        entry = (features["bullish_bos"] | features["bullish_choch"]) & (
            features["swing_trend"] > 0
        ) & (features["volume_zscore_20"] >= 0)
        exit = features["bearish_bos"] | features["bearish_choch"]
        return self._output(features, entry=entry, exit=exit, parameters=p, entry_reason="BULLISH_STRUCTURE_CONTINUATION", exit_reason="BEARISH_STRUCTURE_BREAK")


class BtcRelativeStrengthRotation(Strategy):
    strategy_id = "btc_relative_strength_rotation"
    family = "relative_strength"
    description = "Rotate into an allowed asset when it strengthens relative to BTC."
    defaults = {
        "momentum_threshold": 0.0,
        "stop_atr": 2.0,
        "target_atr": 4.0,
        "trailing_atr": 2.5,
        "maximum_holding_bars": 80,
    }
    parameter_space = {
        "momentum_threshold": (0.0, 0.01, 0.02),
        "stop_atr": (1.5, 2.0, 2.5),
    }

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        momentum = features["btc_relative_momentum_20"]
        entry = crossed_above(momentum, p["momentum_threshold"]) & features["bull_regime"]
        exit = crossed_below(momentum, 0.0) | features["bear_regime"]
        return self._output(features, entry=entry, exit=exit, parameters=p, entry_reason="BTC_RELATIVE_STRENGTH", exit_reason="BTC_RELATIVE_STRENGTH_LOST")


class MultiFactorStrategy(Strategy):
    strategy_id = "multi_factor"
    family = "multi_factor"
    description = "Diversified trend, momentum, volume and structure score."
    defaults = {
        "entry_score": 4,
        "exit_score": 1,
        "stop_atr": 2.0,
        "target_atr": 4.0,
        "trailing_atr": 2.5,
        "maximum_holding_bars": 80,
    }
    parameter_space = {
        "entry_score": (3, 4, 5),
        "exit_score": (1, 2),
        "stop_atr": (1.5, 2.0, 2.5),
    }

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        super().validate_parameters(parameters)
        if parameters["exit_score"] >= parameters["entry_score"]:
            raise ValueError("exit score must be below entry score")

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        score = (
            (features["ema_20"] > features["ema_50"]).astype(int)
            + (features["ema_50"] > features["ema_200"]).astype(int)
            + (features["rsi_14"].between(45, 70)).astype(int)
            + (features["ppo"] > 0).astype(int)
            + (features["volume_zscore_20"] > 0).astype(int)
            + (features["swing_trend"] > 0).astype(int)
        )
        entry = crossed_above(score.astype(float), float(p["entry_score"]) - 0.5)
        exit = crossed_below(score.astype(float), float(p["exit_score"]) + 0.5)
        return self._output(features, entry=entry, exit=exit, parameters=p, entry_reason="MULTI_FACTOR_CONFIRMATION", exit_reason="MULTI_FACTOR_DETERIORATION", metadata={"score": score})


class IntelligenceFilteredTrend(Strategy):
    strategy_id = "intelligence_filtered_trend"
    family = "intelligence_filter"
    description = "Trend entry gated and resized by causal crypto intelligence."
    defaults = {
        "maximum_negative_risk": 0.75,
        "maximum_hack_score": 0.50,
        "stop_atr": 2.0,
        "target_atr": 4.0,
        "trailing_atr": 2.5,
        "maximum_holding_bars": 80,
    }
    parameter_space = {
        "maximum_negative_risk": (0.5, 0.75, 1.0),
        "maximum_hack_score": (0.25, 0.5, 0.75),
    }
    uses_intelligence = True

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        risk = (
            features["negative_risk_event_score"]
            + features["hack_exploit_score"]
            + features["exchange_risk_score"]
        )
        avoid = (
            (features["negative_risk_event_score"] > p["maximum_negative_risk"])
            | (features["hack_exploit_score"] > p["maximum_hack_score"])
        )
        base_entry = (
            features["bull_regime"]
            & crossed_above(features["close"], features["ema_20"])
        )
        exit = features["bear_regime"] | (risk > 2.0)
        sizes = (1.0 - 0.20 * risk).clip(0.25, 1.0)
        reduce = (risk > 1.0) & ~exit
        return self._output(
            features,
            entry=base_entry,
            exit=exit,
            avoid=avoid,
            reduce=reduce,
            size_multiplier=sizes,
            parameters=p,
            entry_reason="INTELLIGENCE_FILTERED_TREND",
            exit_reason="TREND_OR_INTELLIGENCE_RISK_EXIT",
        )


class RiskEventAvoidanceOverlay(Strategy):
    strategy_id = "risk_event_avoidance_overlay"
    family = "risk_overlay"
    description = "No standalone entries; blocks or reduces exposure around risk events."
    defaults = {
        "avoid_threshold": 1.0,
        "exit_threshold": 2.0,
        "stop_atr": 2.0,
        "target_atr": 4.0,
        "trailing_atr": 0.0,
        "maximum_holding_bars": None,
    }
    parameter_space = {
        "avoid_threshold": (0.5, 1.0, 1.5),
        "exit_threshold": (1.5, 2.0, 2.5),
    }
    uses_intelligence = True

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        super().validate_parameters(parameters)
        if parameters["exit_threshold"] <= parameters["avoid_threshold"]:
            raise ValueError("exit threshold must exceed avoid threshold")

    def generate(self, features: pd.DataFrame, parameters: dict[str, Any] | None = None) -> StrategyOutput:
        p = self.parameters(parameters)
        score = (
            features["negative_risk_event_score"]
            + features["hack_exploit_score"]
            + features["exchange_risk_score"]
            + features["stablecoin_risk_score"]
        )
        avoid = score >= p["avoid_threshold"]
        exit = score >= p["exit_threshold"]
        reduce = avoid & ~exit
        sizes = (1.0 - 0.25 * score).clip(0.0, 1.0)
        return self._output(
            features,
            entry=pd.Series(False, index=features.index),
            exit=exit,
            avoid=avoid,
            reduce=reduce,
            size_multiplier=sizes,
            parameters=p,
            entry_reason="OVERLAY_HAS_NO_STANDALONE_ENTRY",
            exit_reason="RISK_EVENT_EXIT",
        )


_STRATEGIES: tuple[Strategy, ...] = (
    EmaTrendPullback(),
    EmaCrossoverTrendFilter(),
    DonchianBreakout(),
    MultiTimeframeDonchianVolumeBreakout(),
    BollingerMeanReversion(),
    RsiMeanReversion(),
    ConnorsRsiPullback(),
    VolatilitySqueezeBreakout(),
    AtrExpansionBreakout(),
    LiquiditySweepReversal(),
    BosChochContinuation(),
    BtcRelativeStrengthRotation(),
    MultiFactorStrategy(),
    IntelligenceFilteredTrend(),
    RiskEventAvoidanceOverlay(),
)


def strategy_registry() -> dict[str, Strategy]:
    return {strategy.strategy_id: strategy for strategy in _STRATEGIES}


def get_strategy(strategy_id: str) -> Strategy:
    try:
        return strategy_registry()[strategy_id]
    except KeyError as exc:
        raise KeyError(f"unknown strategy: {strategy_id}") from exc


def describe_strategies() -> list[dict[str, Any]]:
    return [
        strategy.metadata.model_dump(mode="json")
        | {"defaults": dict(strategy.defaults)}
        for strategy in _STRATEGIES
    ]


__all__ = [
    "Strategy",
    "StrategyOutput",
    "crossed_above",
    "crossed_below",
    "describe_strategies",
    "get_strategy",
    "strategy_registry",
]
