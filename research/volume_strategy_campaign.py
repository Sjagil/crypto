"""Broad, causal candle-volume strategy research across assets and timeframes.

The campaign evaluates only information knowable at a completed candle close
and applies target positions at the next candle open.  It deliberately does
not manufacture CVD, footprint, absorption, VPIN, volume-profile nodes, or
historical order-book signals from OHLCV candles.  Those families remain
blocked until a real timestamped trade/L2 archive exists.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config.settings import Settings
from research.features import volume_features
from research.global_trial_accounting import resolve_known_trial_count
from research.optimization import deflated_sharpe_ratio
from research.portfolio_storm import large_matrix_multiple_testing
from research.strategies import Strategy, StrategyOutput
from research.strategy_registry import (
    ContentAddressedTrialRegistry,
    gaussian_plateau_table,
    plateau_selection_pbo,
)
from utils.common import (
    atomic_write_json,
    read_json,
    sha256_file,
    stable_hash,
)
from utils.pandas_time import sunday_week_end_labels

VOLUME_STRATEGY_CAMPAIGN = "VOLUME_STRATEGY_CATALOG_V1"
VOLUME_STRATEGY_ENGINE_VERSION = "1.0.0"
VOLUME_STRATEGY_FORWARD_START = "2026-07-27T00:00:00+00:00"
VOLUME_STRATEGY_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")
VOLUME_STRATEGY_MARKETS = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "LINK-EUR",
    "ADA-EUR",
    "BNB-EUR",
    "DOGE-EUR",
    "HYPE-EUR",
    "TAO-EUR",
    "TRX-EUR",
    "XRP-EUR",
)
VOLUME_STRATEGY_ALLOWED_MARKETS = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "LINK-EUR",
)
VOLUME_STRATEGY_ARCHETYPES = (
    "DONCHIAN_RVOL_BREAKOUT",
    "TREND_PULLBACK_DRYUP_RECOVERY",
    "VOLUME_CONTRACTION_BREAKOUT",
    "OBV_CMF_CONTINUATION",
    "VWAP_MFI_RECLAIM",
)
VOLUME_STRATEGY_COORDINATES = (0, 1, 2, 3, 4)
PERIODS_PER_YEAR = {
    "5m": 365.25 * 24.0 * 12.0,
    "15m": 365.25 * 24.0 * 4.0,
    "1h": 365.25 * 24.0,
    "4h": 365.25 * 6.0,
    "1d": 365.25,
}

ORDERFLOW_DATA_BLOCKERS = {
    "CVD_DIVERGENCE": "MISSING_HISTORICAL_AGGRESSOR_CLASSIFIED_TRADES",
    "FOOTPRINT_STACKED_IMBALANCE": "MISSING_HISTORICAL_PRICE_LEVEL_TRADE_TAPE",
    "ABSORPTION_REVERSAL": "MISSING_HISTORICAL_TRADE_AND_PRICE_RESPONSE_TAPE",
    "VPIN": "MISSING_HISTORICAL_EQUAL_VOLUME_TRADE_BUCKETS",
    "VOLUME_PROFILE_POC_VAH_VAL_HVN_LVN": (
        "MISSING_HISTORICAL_PRICE_LEVEL_VOLUME_DISTRIBUTION"
    ),
    "ORDER_BOOK_IMBALANCE_MICROPRICE": (
        "MISSING_HISTORICAL_NONCE_VALIDATED_L2_SNAPSHOTS"
    ),
}


@dataclass(frozen=True, slots=True)
class VolumeStrategyDNA:
    """One immutable asset, timeframe, archetype and plateau coordinate."""

    market: str
    timeframe: str
    archetype: str
    coordinate: int
    parameters: Mapping[str, float | int]
    maximum_position_exposure: float = 0.20
    minimum_cash: float = 0.80

    def __post_init__(self) -> None:
        if self.market not in VOLUME_STRATEGY_MARKETS:
            raise ValueError("unsupported volume campaign market")
        if self.timeframe not in VOLUME_STRATEGY_TIMEFRAMES:
            raise ValueError("unsupported volume campaign timeframe")
        if self.archetype not in VOLUME_STRATEGY_ARCHETYPES:
            raise ValueError("unsupported volume strategy archetype")
        if self.coordinate not in VOLUME_STRATEGY_COORDINATES:
            raise ValueError("volume plateau coordinate must be in [0, 4]")
        if self.maximum_position_exposure != 0.20:
            raise ValueError("volume campaign position exposure is fixed at 20%")
        if self.minimum_cash != 0.80:
            raise ValueError("single-sleeve volume campaign cash is fixed at 80%")

    @property
    def strategy_id(self) -> str:
        return (
            f"VOL_{self.market.replace('-', '_')}_{self.timeframe}_"
            f"{self.archetype}_N{self.coordinate}"
        )

    @property
    def group_id(self) -> str:
        return f"{self.market}|{self.timeframe}|{self.archetype}"

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "campaign": VOLUME_STRATEGY_CAMPAIGN,
                "engine_version": VOLUME_STRATEGY_ENGINE_VERSION,
                "market": self.market,
                "timeframe": self.timeframe,
                "archetype": self.archetype,
                "coordinate": self.coordinate,
                "parameters": dict(self.parameters),
                "maximum_position_exposure": self.maximum_position_exposure,
                "minimum_cash": self.minimum_cash,
                "decision": "COMPLETED_CANDLE_CLOSE",
                "execution": "NEXT_OPEN",
            },
            length=64,
        )


@dataclass(frozen=True, slots=True)
class VolumeBacktestBatch:
    returns: pd.DataFrame
    stressed_returns: pd.DataFrame
    gross_returns: pd.DataFrame
    turnover: pd.DataFrame
    positions: pd.DataFrame
    entries: pd.DataFrame
    regimes: pd.DataFrame


def _parameter_path(
    archetype: str,
    coordinate: int,
) -> dict[str, float | int]:
    index = int(coordinate)
    if archetype == "DONCHIAN_RVOL_BREAKOUT":
        return {
            "entry_lookback": (10, 15, 20, 30, 55)[index],
            "exit_lookback": (5, 7, 10, 15, 20)[index],
            "minimum_rvol": (1.00, 1.10, 1.20, 1.35, 1.50)[index],
        }
    if archetype == "TREND_PULLBACK_DRYUP_RECOVERY":
        return {
            "ema_period": (30, 40, 50, 100, 200)[index],
            "maximum_pullback_rvol": (0.90, 0.85, 0.80, 0.75, 0.70)[index],
            "minimum_recovery_rvol": (1.00, 1.10, 1.20, 1.35, 1.50)[index],
        }
    if archetype == "VOLUME_CONTRACTION_BREAKOUT":
        return {
            "channel_lookback": (10, 15, 20, 30, 55)[index],
            "contraction_lookback": (3, 4, 5, 7, 10)[index],
            "maximum_prior_rvol": (0.95, 0.90, 0.85, 0.80, 0.75)[index],
            "minimum_breakout_rvol": (1.00, 1.10, 1.20, 1.35, 1.50)[index],
        }
    if archetype == "OBV_CMF_CONTINUATION":
        return {
            "ema_period": (30, 40, 50, 100, 200)[index],
            "obv_lookback": (5, 10, 15, 20, 30)[index],
            "minimum_cmf": (-0.05, -0.025, 0.0, 0.025, 0.05)[index],
        }
    if archetype == "VWAP_MFI_RECLAIM":
        return {
            "vwap_period": (10, 15, 20, 30, 50)[index],
            "mfi_reclaim": (30.0, 35.0, 40.0, 45.0, 50.0)[index],
            "minimum_rvol": (0.90, 1.00, 1.10, 1.20, 1.35)[index],
        }
    raise ValueError(f"unknown volume strategy archetype: {archetype}")


def volume_strategy_dna(
    available_pairs: tuple[tuple[str, str], ...],
) -> tuple[VolumeStrategyDNA, ...]:
    """Return the complete deterministic five-point plateau search."""

    rows = tuple(
        VolumeStrategyDNA(
            market=market,
            timeframe=timeframe,
            archetype=archetype,
            coordinate=coordinate,
            parameters=_parameter_path(archetype, coordinate),
        )
        for market, timeframe in available_pairs
        for archetype in VOLUME_STRATEGY_ARCHETYPES
        for coordinate in VOLUME_STRATEGY_COORDINATES
    )
    if len({row.dna_hash for row in rows}) != len(rows):
        raise RuntimeError("volume campaign contains duplicate strategy DNA")
    return rows


def _available_paths(settings: Settings) -> dict[tuple[str, str], Path]:
    paths: dict[tuple[str, str], Path] = {}
    for market in VOLUME_STRATEGY_MARKETS:
        for timeframe in VOLUME_STRATEGY_TIMEFRAMES:
            path = (
                settings.paths.processed_data_dir
                / f"{market}_{timeframe}.parquet"
            )
            if path.is_file():
                paths[(market, timeframe)] = path
    return paths


def _microstructure_history_audit(
    settings: Settings,
) -> dict[str, Any]:
    raw_root = settings.paths.raw_data_dir / "bitvavo"
    snapshot_files = tuple(
        (raw_root / "orderbook_snapshot").rglob("*")
    )
    trade_files = tuple((raw_root / "trade").rglob("*"))
    snapshot_files = tuple(path for path in snapshot_files if path.is_file())
    trade_files = tuple(path for path in trade_files if path.is_file())
    database_rows: dict[str, Any] = {}
    if settings.paths.database_path.is_file():
        connection = sqlite3.connect(
            f"file:{settings.paths.database_path.resolve()}?mode=ro",
            uri=True,
        )
        try:
            for table in ("orderbook_statistics", "trades"):
                try:
                    count, first, last = connection.execute(
                        f"SELECT count(*), min(timestamp), max(timestamp) "
                        f"FROM {table}"
                    ).fetchone()
                except sqlite3.OperationalError:
                    count, first, last = 0, None, None
                database_rows[table] = {
                    "rows": int(count),
                    "first_timestamp": first,
                    "last_timestamp": last,
                }
        finally:
            connection.close()
    snapshot_span = database_rows.get(
        "orderbook_statistics",
        {},
    )
    first = snapshot_span.get("first_timestamp")
    last = snapshot_span.get("last_timestamp")
    duration_hours = 0.0
    if first and last:
        duration_hours = (
            pd.Timestamp(last) - pd.Timestamp(first)
        ).total_seconds() / 3_600.0
    return {
        "raw_orderbook_snapshot_files": len(snapshot_files),
        "raw_orderbook_snapshot_bytes": sum(
            path.stat().st_size for path in snapshot_files
        ),
        "raw_trade_batch_files": len(trade_files),
        "raw_trade_batch_bytes": sum(
            path.stat().st_size for path in trade_files
        ),
        "database": database_rows,
        "orderbook_duration_hours": duration_hours,
        "minimum_required_history_days": 90,
        "minimum_required_nonce_validated_snapshots": 100_000,
        "sufficient_for_backtest": bool(
            duration_hours >= 90.0 * 24.0
            and len(snapshot_files) >= 100_000
            and database_rows.get("trades", {}).get("rows", 0)
            >= 100_000
        ),
        "status": "PRESENT_BUT_HISTORICALLY_INSUFFICIENT",
        "synthetic_substitution_permitted": False,
    }


def volume_strategy_campaign_path(settings: Settings) -> Path:
    return (
        settings.paths.lab_dir
        / "reports"
        / "volume_strategy_catalog_campaign_v1.json"
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return _json_ready(value.item())
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def _expected_plan(settings: Settings) -> dict[str, Any]:
    paths = _available_paths(settings)
    pairs = tuple(sorted(paths))
    dna = volume_strategy_dna(pairs)
    allowed_count = sum(
        row.market in VOLUME_STRATEGY_ALLOWED_MARKETS for row in dna
    )
    return {
        "schema_version": "volume_strategy_catalog_plan_v1",
        "status": "CAMPAIGN_PLAN",
        "campaign": VOLUME_STRATEGY_CAMPAIGN,
        "engine_version": VOLUME_STRATEGY_ENGINE_VERSION,
        "economic_hypothesis": (
            "Price continuation and pullback recovery are more credible when "
            "venue-specific spot participation expands in the direction of "
            "the move or contracts during the counter-trend phase."
        ),
        "markets": list(VOLUME_STRATEGY_MARKETS),
        "allowed_promotion_markets": list(
            VOLUME_STRATEGY_ALLOWED_MARKETS
        ),
        "timeframes": list(VOLUME_STRATEGY_TIMEFRAMES),
        "available_market_timeframe_pairs": [
            list(pair) for pair in pairs
        ],
        "archetypes": list(VOLUME_STRATEGY_ARCHETYPES),
        "plateau_coordinates": list(VOLUME_STRATEGY_COORDINATES),
        "plateau_kernel": [0.05, 0.25, 0.40, 0.25, 0.05],
        "trial_count": len(dna),
        "allowed_universe_trial_count": allowed_count,
        "discovery_only_trial_count": len(dna) - allowed_count,
        "strategy_dna_hashes": [row.dna_hash for row in dna],
        "search_space_hash": stable_hash(
            [row.dna_hash for row in dna],
            length=64,
        ),
        "selection_basis": (
            "DEVELOPMENT_ONLY_GAUSSIAN_COMPLETE_FIVE_POINT_PLATEAU"
        ),
        "split_policy": {
            "development": 0.60,
            "validation": 0.20,
            "confirmation": 0.20,
            "chronological": True,
            "validation_used_for_selection": False,
            "confirmation_used_for_selection": False,
        },
        "execution_policy": {
            "signal": "COMPLETED_CANDLE_CLOSE",
            "execution": "NEXT_OPEN",
            "long_only": True,
            "maximum_position_exposure": 0.20,
            "minimum_cash": 0.80,
            "cash_yield": 0.0,
            "normal_costs": "SETTINGS",
            "stressed_cost_multiplier": (
                settings.costs.stressed_cost_multiplier
            ),
        },
        "volume_semantics": {
            "base_volume": "VENUE_SPECIFIC_SPOT_BASE_ASSET_VOLUME",
            "quote_volume": (
                "NATIVE_IF_PRESENT_ELSE_CLOSE_TIMES_BASE_VOLUME_ESTIMATE"
            ),
            "directional_volume": (
                "CANDLE_DIRECTION_PROXY_NOT_TRADE_DELTA_OR_CVD"
            ),
        },
        "orderflow_data_blockers": ORDERFLOW_DATA_BLOCKERS,
        "global_trial_accounting": (
            "EVERY_ASSET_TIMEFRAME_ARCHETYPE_COORDINATE_COUNTS"
        ),
        "promotion_policy": {
            "discovery_assets_can_never_promote": True,
            "allowed_universe_only": True,
            "historical_results_cannot_authorize_live": True,
            "forward_evidence_required": True,
        },
        "ai_development_status": "AI_DEVELOPMENT_EMBARGOED",
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def plan_volume_strategy_campaign(settings: Settings) -> dict[str, Any]:
    """Persist the immutable plan before any result is evaluated."""

    expected = _expected_plan(settings)
    path = (
        settings.paths.lab_dir
        / "reports"
        / "volume_strategy_catalog_plan_v1.json"
    )
    immutable = (
        "campaign",
        "engine_version",
        "markets",
        "allowed_promotion_markets",
        "timeframes",
        "available_market_timeframe_pairs",
        "archetypes",
        "plateau_coordinates",
        "plateau_kernel",
        "trial_count",
        "strategy_dna_hashes",
        "search_space_hash",
        "selection_basis",
        "split_policy",
        "execution_policy",
        "volume_semantics",
        "orderflow_data_blockers",
        "global_trial_accounting",
        "promotion_policy",
    )
    if path.is_file():
        stored = read_json(path)
        for field in immutable:
            if _json_ready(stored.get(field)) != _json_ready(
                expected.get(field)
            ):
                raise RuntimeError(f"VOLUME_STRATEGY_PLAN_DRIFT:{field}")
    else:
        atomic_write_json(path, _json_ready(expected))
    return {
        **expected,
        "plan": str(path),
        "plan_sha256": sha256_file(path),
    }


def _money_flow_index(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (
        frame["high"].astype(float)
        + frame["low"].astype(float)
        + frame["close"].astype(float)
    ) / 3.0
    flow = typical * frame["volume"].astype(float)
    direction = typical.diff()
    positive = flow.where(direction > 0.0, 0.0).rolling(
        period,
        min_periods=period,
    ).sum()
    negative = flow.where(direction < 0.0, 0.0).rolling(
        period,
        min_periods=period,
    ).sum()
    ratio = positive / negative.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + ratio)


def _signals(
    frame: pd.DataFrame,
    rows: tuple[VolumeStrategyDNA, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)
    features = volume_features(frame)
    mfi = _money_flow_index(frame)
    entries: dict[str, pd.Series] = {}
    exits: dict[str, pd.Series] = {}
    for row in rows:
        parameters = row.parameters
        if row.archetype == "DONCHIAN_RVOL_BREAKOUT":
            entry_lookback = int(parameters["entry_lookback"])
            exit_lookback = int(parameters["exit_lookback"])
            prior_high = high.shift(1).rolling(
                entry_lookback,
                min_periods=entry_lookback,
            ).max()
            prior_low = low.shift(1).rolling(
                exit_lookback,
                min_periods=exit_lookback,
            ).min()
            entry = (
                (close > prior_high)
                & (
                    features["relative_volume_20"]
                    >= float(parameters["minimum_rvol"])
                )
            )
            exit_ = close < prior_low
        elif row.archetype == "TREND_PULLBACK_DRYUP_RECOVERY":
            ema_period = int(parameters["ema_period"])
            trend = close.ewm(
                span=ema_period,
                adjust=False,
                min_periods=ema_period,
            ).mean()
            dry_pullback = (
                (close.shift(1) > trend.shift(1) * 0.98)
                & (low.shift(1) <= trend.shift(1) * 1.02)
                & (
                    features["relative_volume_20"].shift(1)
                    <= float(parameters["maximum_pullback_rvol"])
                )
            )
            entry = (
                dry_pullback
                & (close > high.shift(1))
                & (close > trend)
                & (
                    features["relative_volume_20"]
                    >= float(parameters["minimum_recovery_rvol"])
                )
            )
            exit_ = close < trend
        elif row.archetype == "VOLUME_CONTRACTION_BREAKOUT":
            channel = int(parameters["channel_lookback"])
            contraction = int(parameters["contraction_lookback"])
            prior_high = high.shift(1).rolling(
                channel,
                min_periods=channel,
            ).max()
            prior_rvol = features["relative_volume_20"].shift(1).rolling(
                contraction,
                min_periods=contraction,
            ).mean()
            entry = (
                (close > prior_high)
                & (
                    prior_rvol
                    <= float(parameters["maximum_prior_rvol"])
                )
                & (
                    features["relative_volume_20"]
                    >= float(parameters["minimum_breakout_rvol"])
                )
            )
            exit_ = close < close.ewm(
                span=max(10, channel),
                adjust=False,
                min_periods=max(10, channel),
            ).mean()
        elif row.archetype == "OBV_CMF_CONTINUATION":
            ema_period = int(parameters["ema_period"])
            obv_lookback = int(parameters["obv_lookback"])
            trend = close.ewm(
                span=ema_period,
                adjust=False,
                min_periods=ema_period,
            ).mean()
            prior_obv_high = features["obv"].shift(1).rolling(
                obv_lookback,
                min_periods=obv_lookback,
            ).max()
            entry = (
                (close > trend)
                & (features["obv"] > prior_obv_high)
                & (
                    features["chaikin_money_flow_20"]
                    >= float(parameters["minimum_cmf"])
                )
            )
            exit_ = (
                (close < trend)
                | (features["chaikin_money_flow_20"] < 0.0)
            )
        elif row.archetype == "VWAP_MFI_RECLAIM":
            period = int(parameters["vwap_period"])
            typical = (high + low + close) / 3.0
            vwap = (
                (typical * volume).rolling(
                    period,
                    min_periods=period,
                ).sum()
                / volume.rolling(
                    period,
                    min_periods=period,
                ).sum().replace(0.0, np.nan)
            )
            threshold = float(parameters["mfi_reclaim"])
            entry = (
                (close.shift(1) <= vwap.shift(1))
                & (close > vwap)
                & (mfi.shift(1) <= threshold)
                & (mfi > threshold)
                & (
                    features["relative_volume_20"]
                    >= float(parameters["minimum_rvol"])
                )
            )
            exit_ = (close < vwap) | (mfi >= 80.0)
        else:
            raise ValueError(f"unsupported archetype: {row.archetype}")
        entries[row.strategy_id] = entry.fillna(False)
        exits[row.strategy_id] = exit_.fillna(False)
    return (
        pd.DataFrame(entries, index=frame.index, dtype=bool),
        pd.DataFrame(exits, index=frame.index, dtype=bool),
    )


class VolumeCatalogStrategyAdapter(Strategy):
    """Canonical event-driven adapter for one frozen catalog signal DNA.

    The discovery campaign used a vectorized 20%-exposure sleeve without a
    price stop. The canonical engine requires bounded risk. This adapter keeps
    the frozen entry/exit parameters byte-for-byte and adds one explicit,
    separately hashed 10% stop / 30% disaster target safety overlay.
    """

    family = "volume_catalog_canonical_adapter"
    description = "Frozen volume-catalog signals in the canonical backtester."
    parameter_space: dict[str, tuple[Any, ...]] = {}

    def __init__(self, row: VolumeStrategyDNA) -> None:
        self.row = row
        self.strategy_id = row.strategy_id
        self.family = f"volume_catalog_{row.archetype.lower()}"
        self.description = (
            f"Canonical adapter for frozen {row.archetype} coordinate "
            f"{row.coordinate} on {row.market} {row.timeframe}."
        )
        self.defaults = {
            **dict(row.parameters),
            "stop_fraction": 0.10,
            "target_fraction": 0.30,
            "maximum_holding_bars": None,
        }
        self.parameter_space = {
            key: (value,) for key, value in self.defaults.items()
        }
        self.legacy_strategy_dna_hash = row.dna_hash
        self.canonical_adapter_dna_hash = stable_hash(
            {
                "legacy_strategy_dna_hash": row.dna_hash,
                "adapter": "CANONICAL_BOUNDED_RISK_V1",
                "stop_fraction": 0.10,
                "target_fraction": 0.30,
            },
            length=64,
        )
        self.material_difference_reason = (
            "LEGACY_20PCT_UNBOUNDED_SLEEVE_REPLACED_BY_CANONICAL_"
            "RISK_SIZING_WITH_10PCT_STOP_AND_30PCT_DISASTER_TARGET"
        )

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        if not 0 < float(parameters["stop_fraction"]) < 1:
            raise ValueError("volume adapter stop fraction must be in (0,1)")
        if float(parameters["target_fraction"]) <= 0:
            raise ValueError("volume adapter target fraction must be positive")

    def generate(
        self,
        features: pd.DataFrame,
        parameters: dict[str, Any] | None = None,
    ) -> StrategyOutput:
        selected = self.parameters(parameters)
        frozen = VolumeStrategyDNA(
            market=self.row.market,
            timeframe=self.row.timeframe,
            archetype=self.row.archetype,
            coordinate=self.row.coordinate,
            parameters={
                key: selected[key] for key in self.row.parameters
            },
        )
        entries, exits = _signals(features, (frozen,))
        entry = entries[frozen.strategy_id]
        exit_ = exits[frozen.strategy_id]
        close = features["close"].astype(float)
        return StrategyOutput(
            entry=(entry & ~exit_).fillna(False).astype(bool),
            exit=exit_.fillna(False).astype(bool),
            avoid=pd.Series(False, index=features.index),
            reduce=pd.Series(False, index=features.index),
            stop_distance=close * float(selected["stop_fraction"]),
            target_distance=close * float(selected["target_fraction"]),
            trailing_distance=pd.Series(0.0, index=features.index),
            size_multiplier=pd.Series(1.0, index=features.index),
            maximum_holding_bars=None,
            entry_reason=f"FROZEN_{frozen.archetype}_ENTRY",
            exit_reason=f"FROZEN_{frozen.archetype}_EXIT",
            metadata={
                "legacy_strategy_dna_hash": frozen.dna_hash,
                "canonical_adapter_dna_hash": self.canonical_adapter_dna_hash,
                "material_difference_reason": self.material_difference_reason,
                "legacy_parameters_unchanged": True,
            },
        ).validate(features.index)


@lru_cache(maxsize=256)
def volume_strategy_adapter(strategy_id: str) -> VolumeCatalogStrategyAdapter:
    """Resolve an exact frozen catalog strategy ID into its safe adapter."""

    available_pairs = tuple(
        (market, timeframe)
        for market in VOLUME_STRATEGY_MARKETS
        for timeframe in VOLUME_STRATEGY_TIMEFRAMES
    )
    matches = [
        row
        for row in volume_strategy_dna(available_pairs)
        if row.strategy_id == strategy_id
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown volume catalog strategy: {strategy_id}")
    return VolumeCatalogStrategyAdapter(matches[0])


def _regime_labels(
    frame: pd.DataFrame,
    btc_frame: pd.DataFrame,
    *,
    timeframe: str,
) -> pd.DataFrame:
    index = frame.index
    btc_close = btc_frame["close"].astype(float).reindex(index).ffill()
    btc_ema = btc_close.ewm(
        span=200,
        adjust=False,
        min_periods=200,
    ).mean()
    btc_slope = btc_ema.pct_change(20, fill_method=None)
    phase = np.select(
        [
            (btc_close > btc_ema) & (btc_slope > 0.0),
            (btc_close < btc_ema) & (btc_slope < 0.0),
        ],
        ["TREND_UP", "TREND_DOWN"],
        default="RANGE_OR_TRANSITION",
    )
    btc_returns = np.log(btc_close.where(btc_close > 0.0)).diff()
    realized = btc_returns.rolling(
        30,
        min_periods=30,
    ).std(ddof=0)
    baseline = realized.expanding(min_periods=252).median().shift(1)
    volatility = np.where(
        realized >= baseline,
        "HIGH",
        "LOW",
    )
    asset_close = frame["close"].astype(float)
    asset_ema = asset_close.ewm(
        span=200,
        adjust=False,
        min_periods=200,
    ).mean()
    asset_trend = np.where(
        asset_close >= asset_ema,
        "UP",
        "DOWN",
    )
    rvol = volume_features(frame)["relative_volume_20"]
    participation = np.where(
        rvol >= 1.20,
        "EXPANSION",
        "NORMAL_OR_DRY",
    )
    if timeframe == "1d":
        session = np.full(len(index), "ALL_DAY", dtype=object)
    else:
        hours = index.hour
        session = np.select(
            [hours < 8, hours < 16],
            ["ASIA_UTC", "EUROPE_UTC"],
            default="US_UTC",
        )
    return pd.DataFrame(
        {
            "btc_phase": phase,
            "btc_volatility": volatility,
            "asset_trend": asset_trend,
            "participation": participation,
            "session": session,
        },
        index=index,
    )


def backtest_volume_strategy_batch(
    frame: pd.DataFrame,
    btc_frame: pd.DataFrame,
    rows: tuple[VolumeStrategyDNA, ...],
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    stressed_cost_multiplier: float,
) -> VolumeBacktestBatch:
    """Evaluate one market/timeframe batch with next-open execution."""

    if not rows:
        raise ValueError("volume backtest batch cannot be empty")
    if len({(row.market, row.timeframe) for row in rows}) != 1:
        raise ValueError("volume batch must contain one market/timeframe")
    selected = frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    selected.index = pd.to_datetime(selected.index, utc=True)
    selected = selected[
        ~selected.index.duplicated(keep="last")
    ].sort_index()
    if len(selected) < 500:
        raise ValueError("volume strategy batch requires at least 500 candles")
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("volume strategy batch contains non-finite OHLCV")
    if bool(
        (
            selected.loc[:, ["open", "high", "low", "close"]]
            .to_numpy(dtype=float)
            <= 0.0
        ).any()
    ):
        raise ValueError("volume strategy batch contains non-positive prices")
    if bool((selected["volume"].to_numpy(dtype=float) < 0.0).any()):
        raise ValueError("volume strategy batch contains negative volume")

    entries, exits = _signals(selected, rows)
    entry_values = entries.to_numpy(dtype=bool)
    exit_values = exits.to_numpy(dtype=bool)
    state = np.zeros(len(rows), dtype=bool)
    targets = np.zeros(entry_values.shape, dtype=np.float32)
    for position in range(len(selected)):
        state &= ~exit_values[position]
        state |= entry_values[position] & ~exit_values[position]
        targets[position] = state
    executed = np.zeros_like(targets)
    executed[1:] = targets[:-1]
    exposure = executed * 0.20

    opens = selected["open"].to_numpy(dtype=float)
    closes = selected["close"].to_numpy(dtype=float)
    underlying = np.empty(len(selected) - 1, dtype=float)
    underlying[:-1] = opens[2:] / opens[1:-1] - 1.0
    underlying[-1] = closes[-1] / opens[-1] - 1.0
    interval_exposure = exposure[1:]
    gross = interval_exposure * underlying[:, None]
    turnover = np.abs(
        np.diff(
            exposure,
            axis=0,
        )
    )
    turnover[-1] += exposure[-1]
    one_way = (
        fee_rate
        + slippage_bps / 10_000.0
        + spread_bps / 20_000.0
    )
    costs = turnover * one_way
    stressed_costs = turnover * one_way * stressed_cost_multiplier
    net = (1.0 - costs) * (1.0 + gross) - 1.0
    stressed = (
        (1.0 - stressed_costs) * (1.0 + gross) - 1.0
    )
    if bool((costs >= 1.0).any()) or bool(
        (stressed_costs >= 1.0).any()
    ):
        raise ValueError("volume strategy costs exhaust portfolio equity")
    output_index = selected.index[1:]
    columns = [row.strategy_id for row in rows]
    regimes = _regime_labels(
        selected,
        btc_frame,
        timeframe=rows[0].timeframe,
    ).reindex(output_index)
    return VolumeBacktestBatch(
        returns=pd.DataFrame(net, index=output_index, columns=columns),
        stressed_returns=pd.DataFrame(
            stressed,
            index=output_index,
            columns=columns,
        ),
        gross_returns=pd.DataFrame(
            gross,
            index=output_index,
            columns=columns,
        ),
        turnover=pd.DataFrame(
            turnover,
            index=output_index,
            columns=columns,
        ),
        positions=pd.DataFrame(
            interval_exposure,
            index=output_index,
            columns=columns,
        ),
        entries=entries.iloc[1:].copy(),
        regimes=regimes,
    )


def _period_metrics(
    returns: pd.Series,
    turnover: pd.Series,
    positions: pd.Series,
    *,
    periods_per_year: float,
) -> dict[str, Any]:
    values = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {
            "observations": 0,
            "net_return": 0.0,
            "sharpe": 0.0,
            "maximum_drawdown": 0.0,
            "profit_factor": 0.0,
            "trade_entries": 0,
            "average_exposure": 0.0,
        }
    equity = (1.0 + values).cumprod()
    standard = float(values.std(ddof=1))
    positive = float(values[values > 0.0].sum())
    negative = abs(float(values[values < 0.0].sum()))
    aligned_turnover = turnover.reindex(values.index).fillna(0.0)
    aligned_positions = positions.reindex(values.index).fillna(0.0)
    entries = (
        (aligned_positions > 0.0)
        & (aligned_positions.shift(1, fill_value=0.0) <= 0.0)
    )
    return {
        "observations": int(len(values)),
        "net_return": float(equity.iloc[-1] - 1.0),
        "sharpe": (
            float(
                values.mean()
                / standard
                * math.sqrt(periods_per_year)
            )
            if standard > 0.0
            else 0.0
        ),
        "maximum_drawdown": float(
            (equity / equity.cummax() - 1.0).min()
        ),
        "profit_factor": (
            positive / negative
            if negative > 0.0
            else (math.inf if positive > 0.0 else 0.0)
        ),
        "trade_entries": int(entries.sum()),
        "turnover": float(aligned_turnover.sum()),
        "average_exposure": float(aligned_positions.mean()),
        "positive_periods": int((values > 0.0).sum()),
        "negative_periods": int((values < 0.0).sum()),
    }


def _split_slices(length: int) -> dict[str, slice]:
    development_end = max(1, int(length * 0.60))
    validation_end = max(development_end + 1, int(length * 0.80))
    validation_end = min(validation_end, length - 1)
    return {
        "development": slice(0, development_end),
        "validation": slice(development_end, validation_end),
        "confirmation": slice(validation_end, length),
    }


def _regime_rows(
    strategy_id: str,
    returns: pd.Series,
    regimes: pd.DataFrame,
    *,
    periods_per_year: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aligned = regimes.reindex(returns.index)
    for axis in aligned.columns:
        for state in sorted(aligned[axis].dropna().astype(str).unique()):
            selected = returns.loc[aligned[axis].astype(str) == state]
            if len(selected) < 30:
                continue
            standard = float(selected.std(ddof=1))
            positive = float(selected[selected > 0.0].sum())
            negative = abs(float(selected[selected < 0.0].sum()))
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "axis": axis,
                    "state": state,
                    "observations": int(len(selected)),
                    "compound_return": float(
                        (1.0 + selected).prod() - 1.0
                    ),
                    "mean_return": float(selected.mean()),
                    "sharpe": (
                        float(
                            selected.mean()
                            / standard
                            * math.sqrt(periods_per_year)
                        )
                        if standard > 0.0
                        else 0.0
                    ),
                    "profit_factor": (
                        positive / negative
                        if negative > 0.0
                        else (
                            math.inf if positive > 0.0 else 0.0
                        )
                    ),
                }
            )
    return rows


def _weekly(series: pd.Series) -> pd.Series:
    selected = 1.0 + series
    return selected.groupby(sunday_week_end_labels(selected.index)).prod().sub(
        1.0
    ).replace([np.inf, -np.inf], np.nan).dropna()


def _selection_summary(
    primary_id: str,
    summary: pd.DataFrame,
    weekly: pd.DataFrame,
    stressed_weekly: pd.DataFrame,
    *,
    total_known_trials: int,
    trial_sharpes: list[float],
    settings: Settings,
    pbo: float | None,
    multiple_testing: Mapping[str, Any],
) -> dict[str, Any]:
    row = summary.loc[primary_id].to_dict()
    confirmation_start = int(len(weekly) * 0.80)
    confirmation = weekly[primary_id].iloc[confirmation_start:]
    stressed_confirmation = stressed_weekly[primary_id].iloc[
        confirmation_start:
    ]
    dsr = deflated_sharpe_ratio(
        weekly[primary_id].iloc[: int(len(weekly) * 0.60)],
        trial_sharpes,
        total_trials=total_known_trials,
    )
    checks = {
        "plateau_eligible": bool(row["plateau_eligible"]),
        "validation_positive": float(
            row["validation_net_return"]
        ) > 0.0,
        "confirmation_positive": float(
            row["confirmation_net_return"]
        ) > 0.0,
        "stressed_confirmation_positive": (
            float(row["stressed_confirmation_net_return"]) > 0.0
        ),
        "full_profit_factor": (
            float(row["full_profit_factor"])
            >= settings.research.minimum_profit_factor
        ),
        "minimum_trades": (
            int(row["full_trade_entries"])
            >= settings.research.minimum_trades
        ),
        "maximum_drawdown": (
            abs(float(row["full_maximum_drawdown"]))
            <= settings.research.maximum_drawdown
        ),
        "deflated_sharpe": (
            dsr
            >= settings.research.minimum_deflated_sharpe_probability
        ),
        "white_reality_check": (
            float(
                multiple_testing["white_reality_check_pvalue"]
            )
            <= settings.research.maximum_white_reality_check_pvalue
        ),
        "hansen_spa": (
            float(multiple_testing["hansen_spa_pvalue"])
            <= settings.research.maximum_hansen_spa_pvalue
        ),
        "pbo": (
            pbo is not None
            and pbo
            <= settings.research.maximum_probability_of_backtest_overfitting
        ),
        "untouched_holdout": False,
        "forward_evidence": False,
    }
    return {
        "strategy_id": primary_id,
        "strategy_dna_hash": row["strategy_dna_hash"],
        "market": row["market"],
        "timeframe": row["timeframe"],
        "archetype": row["archetype"],
        "coordinate": int(row["coordinate"]),
        "parameters": row["parameters"],
        "metrics": row,
        "confirmation_weekly_observations": int(len(confirmation)),
        "confirmation_weekly_net_return_reconciled": float(
            (1.0 + confirmation).prod() - 1.0
        ),
        "stressed_confirmation_weekly_net_return_reconciled": float(
            (1.0 + stressed_confirmation).prod() - 1.0
        ),
        "deflated_sharpe_probability": dsr,
        "checks": checks,
        "economic_pass": all(
            checks[name]
            for name in (
                "plateau_eligible",
                "validation_positive",
                "confirmation_positive",
                "stressed_confirmation_positive",
                "full_profit_factor",
                "minimum_trades",
                "maximum_drawdown",
            )
        ),
        "statistical_pass": all(
            checks[name]
            for name in (
                "deflated_sharpe",
                "white_reality_check",
                "hansen_spa",
                "pbo",
            )
        ),
        "research_pass": False,
        "paper_candidate_permitted": False,
        "orders_generated": 0,
        "live_ready": False,
    }


def run_volume_strategy_campaign(settings: Settings) -> dict[str, Any]:
    """Run every preregistered real-OHLCV volume strategy path."""

    plan = plan_volume_strategy_campaign(settings)
    paths = _available_paths(settings)
    pairs = tuple(sorted(paths))
    all_dna = volume_strategy_dna(pairs)
    by_pair = {
        pair: tuple(
            row
            for row in all_dna
            if (row.market, row.timeframe) == pair
        )
        for pair in pairs
    }
    data_hashes = {
        f"{market}|{timeframe}": sha256_file(path)
        for (market, timeframe), path in paths.items()
    }
    data_fingerprint = stable_hash(data_hashes, length=64)
    frames = {
        pair: pd.read_parquet(path)
        for pair, path in paths.items()
    }
    registry = ContentAddressedTrialRegistry(
        settings.paths.lab_dir
        / "strategy_registry"
        / "volume_strategy_catalog_v1",
        campaign_id=VOLUME_STRATEGY_CAMPAIGN,
    )
    summary_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    weekly_paths: dict[str, pd.Series] = {}
    stressed_weekly_paths: dict[str, pd.Series] = {}
    dna_by_id = {row.strategy_id: row for row in all_dna}

    for pair in pairs:
        market, timeframe = pair
        btc_pair = ("BTC-EUR", timeframe)
        if btc_pair not in frames:
            raise RuntimeError(f"BTC benchmark missing for {timeframe}")
        rows = by_pair[pair]
        batch = backtest_volume_strategy_batch(
            frames[pair],
            frames[btc_pair],
            rows,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            stressed_cost_multiplier=(
                settings.costs.stressed_cost_multiplier
            ),
        )
        slices = _split_slices(len(batch.returns))
        for row in rows:
            strategy_id = row.strategy_id
            periods: dict[str, Any] = {}
            stressed_periods: dict[str, Any] = {}
            for name, selected_slice in slices.items():
                periods[name] = _period_metrics(
                    batch.returns[strategy_id].iloc[selected_slice],
                    batch.turnover[strategy_id].iloc[selected_slice],
                    batch.positions[strategy_id].iloc[selected_slice],
                    periods_per_year=PERIODS_PER_YEAR[timeframe],
                )
                stressed_periods[name] = _period_metrics(
                    batch.stressed_returns[strategy_id].iloc[
                        selected_slice
                    ],
                    batch.turnover[strategy_id].iloc[selected_slice],
                    batch.positions[strategy_id].iloc[selected_slice],
                    periods_per_year=PERIODS_PER_YEAR[timeframe],
                )
            full = _period_metrics(
                batch.returns[strategy_id],
                batch.turnover[strategy_id],
                batch.positions[strategy_id],
                periods_per_year=PERIODS_PER_YEAR[timeframe],
            )
            stressed_full = _period_metrics(
                batch.stressed_returns[strategy_id],
                batch.turnover[strategy_id],
                batch.positions[strategy_id],
                periods_per_year=PERIODS_PER_YEAR[timeframe],
            )
            summary_rows.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_dna_hash": row.dna_hash,
                    "market": market,
                    "timeframe": timeframe,
                    "universe_role": (
                        "ALLOWED_PROMOTION_UNIVERSE"
                        if market in VOLUME_STRATEGY_ALLOWED_MARKETS
                        else "DISCOVERY_ONLY"
                    ),
                    "archetype": row.archetype,
                    "coordinate": row.coordinate,
                    "parameters": dict(row.parameters),
                    "group_id": row.group_id,
                    "development_net_return": periods["development"][
                        "net_return"
                    ],
                    "development_sharpe": periods["development"][
                        "sharpe"
                    ],
                    "development_profit_factor": periods[
                        "development"
                    ]["profit_factor"],
                    "validation_net_return": periods["validation"][
                        "net_return"
                    ],
                    "validation_sharpe": periods["validation"][
                        "sharpe"
                    ],
                    "confirmation_net_return": periods[
                        "confirmation"
                    ]["net_return"],
                    "confirmation_sharpe": periods["confirmation"][
                        "sharpe"
                    ],
                    "stressed_confirmation_net_return": (
                        stressed_periods["confirmation"]["net_return"]
                    ),
                    "full_net_return": full["net_return"],
                    "full_sharpe": full["sharpe"],
                    "full_profit_factor": full["profit_factor"],
                    "full_maximum_drawdown": full[
                        "maximum_drawdown"
                    ],
                    "full_trade_entries": full["trade_entries"],
                    "full_average_exposure": full[
                        "average_exposure"
                    ],
                    "stressed_full_net_return": stressed_full[
                        "net_return"
                    ],
                    "orders_generated": 0,
                    "live_ready": False,
                }
            )
            weekly_paths[strategy_id] = _weekly(
                batch.returns[strategy_id]
            )
            stressed_weekly_paths[strategy_id] = _weekly(
                batch.stressed_returns[strategy_id]
            )
            regime_rows.extend(
                _regime_rows(
                    strategy_id,
                    batch.returns[strategy_id],
                    batch.regimes,
                    periods_per_year=PERIODS_PER_YEAR[timeframe],
                )
            )
            development_slice = slices["development"]
            development_returns = batch.returns[strategy_id].iloc[
                development_slice
            ]
            registry.register(
                data_fingerprint=data_hashes[
                    f"{market}|{timeframe}"
                ],
                strategy_family=row.archetype,
                strategy_dna_hash=row.dna_hash,
                parameters={
                    **asdict(row),
                    "parameters": dict(row.parameters),
                },
                metrics_at_birth={
                    **periods["development"],
                    "selection_basis": (
                        "DEVELOPMENT_ONLY_GAUSSIAN_PLATEAU"
                    ),
                },
                return_path_hash=stable_hash(
                    [
                        round(float(value), 15)
                        for value in development_returns.to_numpy(
                            dtype=float
                        )
                    ],
                    length=64,
                ),
                selection_metadata={
                    "development_only": True,
                    "validation_used": False,
                    "confirmation_used": False,
                    "plateau_coordinate": row.coordinate,
                    "plateau_group": row.group_id,
                },
            )

    summary = pd.DataFrame(summary_rows).set_index("strategy_id")
    weekly = pd.concat(
        weekly_paths,
        axis=1,
        join="inner",
    ).dropna(how="any")
    stressed_weekly = pd.concat(
        stressed_weekly_paths,
        axis=1,
        join="inner",
    ).dropna(how="any")
    if len(weekly) < 20:
        raise RuntimeError("volume campaign common weekly history too short")
    development_end = max(8, int(len(weekly) * 0.60))
    development = weekly.iloc[:development_end]
    coordinates = {
        strategy_id: dna_by_id[strategy_id].coordinate
        for strategy_id in weekly.columns
    }
    groups = {
        strategy_id: dna_by_id[strategy_id].group_id
        for strategy_id in weekly.columns
    }
    plateau = gaussian_plateau_table(
        development,
        coordinates=coordinates,
        groups=groups,
    )
    summary = summary.join(
        plateau[
            [
                "complete_neighborhood",
                "gaussian_smoothed_sharpe",
                "minimum_neighbor_sharpe",
                "all_neighbors_net_positive",
                "plateau_eligible",
            ]
        ],
        how="left",
    )
    plateau_pbo, plateau_logits = plateau_selection_pbo(
        development,
        coordinates=coordinates,
        groups=groups,
    )
    allowed_ids = [
        strategy_id
        for strategy_id in summary.index
        if summary.loc[strategy_id, "market"]
        in VOLUME_STRATEGY_ALLOWED_MARKETS
    ]
    allowed_development = development.loc[:, allowed_ids]
    allowed_coordinates = {
        strategy_id: coordinates[strategy_id]
        for strategy_id in allowed_ids
    }
    allowed_groups = {
        strategy_id: groups[strategy_id]
        for strategy_id in allowed_ids
    }
    allowed_pbo, allowed_pbo_logits = plateau_selection_pbo(
        allowed_development,
        coordinates=allowed_coordinates,
        groups=allowed_groups,
    )
    allowed_plateau = summary.loc[allowed_ids]
    eligible = allowed_plateau[
        allowed_plateau["plateau_eligible"].fillna(False).astype(bool)
    ]
    selection_pool = (
        eligible
        if not eligible.empty
        else allowed_plateau[
            allowed_plateau["coordinate"] == 2
        ]
    )
    primary_id = str(
        selection_pool["gaussian_smoothed_sharpe"].astype(float).idxmax()
    )

    registry_audit = registry.audit()
    total_known_trials = resolve_known_trial_count(
        settings.paths.lab_dir,
        local_known_trial_count=int(
            registry_audit["unique_strategy_dna_count"]
        ),
    )
    multiple = large_matrix_multiple_testing(
        development.to_numpy(dtype=float),
        bootstrap_samples=(
            settings.research.multiple_testing_bootstrap_samples
        ),
        block_size=max(
            2,
            settings.research.multiple_testing_block_size,
        ),
        seed=settings.app.random_seed + 310_000,
        batch_size=32,
    )
    allowed_multiple = large_matrix_multiple_testing(
        allowed_development.to_numpy(dtype=float),
        bootstrap_samples=(
            settings.research.multiple_testing_bootstrap_samples
        ),
        block_size=max(
            2,
            settings.research.multiple_testing_block_size,
        ),
        seed=settings.app.random_seed + 310_001,
        batch_size=32,
    )
    multiple["plateau_selection_pbo"] = plateau_pbo
    multiple["plateau_selection_pbo_logits"] = list(plateau_logits)
    allowed_multiple["plateau_selection_pbo"] = allowed_pbo
    allowed_multiple["plateau_selection_pbo_logits"] = list(
        allowed_pbo_logits
    )
    timeframe_audits: dict[str, Any] = {}
    timeframe_audit_rows: list[dict[str, Any]] = []
    for timeframe_index, timeframe in enumerate(
        VOLUME_STRATEGY_TIMEFRAMES
    ):
        cohort_ids = [
            strategy_id
            for strategy_id in allowed_ids
            if dna_by_id[strategy_id].timeframe == timeframe
        ]
        if len(cohort_ids) < 2:
            continue
        cohort_weekly = pd.concat(
            {
                strategy_id: weekly_paths[strategy_id]
                for strategy_id in cohort_ids
            },
            axis=1,
            join="inner",
        ).dropna(how="any")
        cohort_development_end = max(
            8,
            int(len(cohort_weekly) * 0.60),
        )
        cohort_development = cohort_weekly.iloc[
            :cohort_development_end
        ]
        cohort_coordinates = {
            strategy_id: coordinates[strategy_id]
            for strategy_id in cohort_ids
        }
        cohort_groups = {
            strategy_id: groups[strategy_id]
            for strategy_id in cohort_ids
        }
        cohort_plateau = gaussian_plateau_table(
            cohort_development,
            coordinates=cohort_coordinates,
            groups=cohort_groups,
        )
        cohort_eligible = cohort_plateau[
            cohort_plateau["plateau_eligible"].astype(bool)
        ]
        cohort_center = cohort_plateau[
            cohort_plateau["coordinate"] == 2
        ]
        diagnostic_pool = (
            cohort_eligible
            if not cohort_eligible.empty
            else cohort_center
        )
        diagnostic_id = str(
            diagnostic_pool["gaussian_smoothed_sharpe"].idxmax()
        )
        cohort_pbo, cohort_logits = plateau_selection_pbo(
            cohort_development,
            coordinates=cohort_coordinates,
            groups=cohort_groups,
        )
        cohort_multiple = large_matrix_multiple_testing(
            cohort_development.to_numpy(dtype=float),
            bootstrap_samples=(
                settings.research.multiple_testing_bootstrap_samples
            ),
            block_size=max(
                2,
                settings.research.multiple_testing_block_size,
            ),
            seed=(
                settings.app.random_seed
                + 311_000
                + timeframe_index
            ),
            batch_size=32,
        )
        cohort_audit = {
            "timeframe": timeframe,
            "strategy_count": len(cohort_ids),
            "full_common_weekly_observations": int(
                len(cohort_weekly)
            ),
            "development_weekly_observations": int(
                len(cohort_development)
            ),
            "sample_sufficiency": (
                "ADEQUATE_DIAGNOSTIC"
                if len(cohort_development) >= 52
                else "LIMITED_UNDER_52_DEVELOPMENT_WEEKS"
            ),
            "plateau_eligible_count": int(len(cohort_eligible)),
            "diagnostic_primary_strategy_id": diagnostic_id,
            "diagnostic_primary_plateau_eligible": bool(
                cohort_plateau.loc[
                    diagnostic_id,
                    "plateau_eligible",
                ]
            ),
            "diagnostic_primary_smoothed_sharpe": float(
                cohort_plateau.loc[
                    diagnostic_id,
                    "gaussian_smoothed_sharpe",
                ]
            ),
            "diagnostic_primary_minimum_neighbor_sharpe": float(
                cohort_plateau.loc[
                    diagnostic_id,
                    "minimum_neighbor_sharpe",
                ]
            ),
            "plateau_selection_pbo": cohort_pbo,
            "plateau_selection_pbo_logits": list(cohort_logits),
            "white_reality_check_pvalue": cohort_multiple[
                "white_reality_check_pvalue"
            ],
            "hansen_spa_pvalue": cohort_multiple[
                "hansen_spa_pvalue"
            ],
            "ordinary_winner_pbo": cohort_multiple[
                "probability_of_backtest_overfitting"
            ],
            "selection_authority": "DIAGNOSTIC_ONLY_NO_RESELECTION",
        }
        timeframe_audits[timeframe] = cohort_audit
        timeframe_audit_rows.append(
            {
                key: value
                for key, value in cohort_audit.items()
                if key != "plateau_selection_pbo_logits"
            }
        )
    trial_sharpes = (
        summary["development_sharpe"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
        .tolist()
    )
    primary = _selection_summary(
        primary_id,
        summary,
        weekly,
        stressed_weekly,
        total_known_trials=total_known_trials,
        trial_sharpes=trial_sharpes,
        settings=settings,
        pbo=allowed_pbo,
        multiple_testing=allowed_multiple,
    )
    discovery_pool = summary[
        summary["plateau_eligible"].fillna(False).astype(bool)
    ]
    discovery_primary_id = str(
        (
            discovery_pool
            if not discovery_pool.empty
            else summary[summary["coordinate"] == 2]
        )["gaussian_smoothed_sharpe"].astype(float).idxmax()
    )
    regime_frame = pd.DataFrame(regime_rows)
    primary_regimes = (
        regime_frame[
            regime_frame["strategy_id"] == primary_id
        ]
        .sort_values(
            ["axis", "sharpe"],
            ascending=[True, False],
        )
        .to_dict(orient="records")
    )
    best_regimes = (
        regime_frame.sort_values(
            ["strategy_id", "axis", "sharpe"],
            ascending=[True, True, False],
        )
        .groupby(["strategy_id", "axis"], as_index=False)
        .first()
    )

    report_path = volume_strategy_campaign_path(settings)
    csv_path = report_path.with_suffix(".csv")
    regimes_path = report_path.with_name(
        "volume_strategy_catalog_regimes_v1.csv"
    )
    timeframe_audit_path = report_path.with_name(
        "volume_strategy_catalog_timeframe_audit_v1.csv"
    )
    csv_ready = summary.copy()
    csv_ready["parameters"] = csv_ready["parameters"].map(
        lambda value: stable_hash(value, length=16)
    )
    csv_ready.reset_index().to_csv(csv_path, index=False)
    regime_frame.to_csv(regimes_path, index=False)
    pd.DataFrame(timeframe_audit_rows).to_csv(
        timeframe_audit_path,
        index=False,
    )
    observer_path = (
        settings.paths.lab_dir
        / "observers"
        / "volume_strategy_catalog_v1"
        / f"{primary_id.lower()}.json"
    )
    observer = {
        "schema_version": "volume_strategy_forward_observer_v1",
        "status": "FROZEN_FORWARD_RESEARCH",
        "campaign": VOLUME_STRATEGY_CAMPAIGN,
        "strategy_id": primary_id,
        "strategy_dna_hash": primary["strategy_dna_hash"],
        "market": primary["market"],
        "timeframe": primary["timeframe"],
        "forward_start": VOLUME_STRATEGY_FORWARD_START,
        "minimum_forward_calendar_days": 365,
        "minimum_closed_strategy_bars": (
            int(PERIODS_PER_YEAR[primary["timeframe"]])
        ),
        "minimum_closed_trades": 30,
        "parameters_frozen": True,
        "orders_generated": 0,
        "orders_submitted": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    atomic_write_json(observer_path, observer)

    payload = {
        "schema_version": "volume_strategy_catalog_report_v1",
        "status": "COMPLETED_NOT_PROMOTED",
        "campaign": VOLUME_STRATEGY_CAMPAIGN,
        "engine_version": VOLUME_STRATEGY_ENGINE_VERSION,
        "plan": plan["plan"],
        "plan_sha256": plan["plan_sha256"],
        "search_space_hash": plan["search_space_hash"],
        "selection_basis": plan["selection_basis"],
        "generated_trial_count": len(all_dna),
        "registered_unique_trials": int(
            registry_audit["unique_strategy_dna_count"]
        ),
        "registered_epoch_records": int(
            registry_audit["unique_epoch_record_count"]
        ),
        "total_known_trials": total_known_trials,
        "market_timeframe_pairs": len(pairs),
        "allowed_universe_trials": int(
            (
                summary["universe_role"]
                == "ALLOWED_PROMOTION_UNIVERSE"
            ).sum()
        ),
        "discovery_only_trials": int(
            (summary["universe_role"] == "DISCOVERY_ONLY").sum()
        ),
        "common_weekly_observations": int(len(weekly)),
        "development_weekly_observations": int(len(development)),
        "primary_allowed_universe_result": primary,
        "discovery_primary_strategy_id": discovery_primary_id,
        "discovery_primary_is_promotable": False,
        "primary_regime_performance": primary_regimes,
        "best_regime_by_strategy_axis": best_regimes.to_dict(
            orient="records"
        ),
        "multiple_testing_all_discovery": multiple,
        "multiple_testing_allowed_only": allowed_multiple,
        "timeframe_cohort_audits": timeframe_audits,
        "timeframe_cohort_audit_policy": (
            "DIAGNOSTIC_ONLY_PRESERVES_ORIGINAL_GLOBAL_SELECTION"
        ),
        "trial_registry": registry_audit,
        "data_fingerprint": data_fingerprint,
        "data_hashes": data_hashes,
        "volume_semantics": plan["volume_semantics"],
        "orderflow_data_blockers": ORDERFLOW_DATA_BLOCKERS,
        "historical_orderflow_archive": {
            **_microstructure_history_audit(settings),
            "orderflow_strategies_backtested": 0,
            "reason": "REAL_HISTORY_REQUIRED_NO_SYNTHETIC_SUBSTITUTE",
        },
        "execution_policy": plan["execution_policy"],
        "promotion_policy": plan["promotion_policy"],
        "holdout_status": (
            "NO_GLOBALLY_UNTOUCHED_HISTORICAL_HOLDOUT_REMAINS"
        ),
        "forward_observer": str(observer_path),
        "research_pass": False,
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
        "ai_development_status": "AI_DEVELOPMENT_EMBARGOED",
    }
    atomic_write_json(report_path, _json_ready(payload))
    return {
        "campaign": VOLUME_STRATEGY_CAMPAIGN,
        "status": payload["status"],
        "report": str(report_path),
        "csv": str(csv_path),
        "regimes_csv": str(regimes_path),
        "timeframe_audit_csv": str(timeframe_audit_path),
        "plan": plan["plan"],
        "generated_trial_count": len(all_dna),
        "registered_unique_trials": payload[
            "registered_unique_trials"
        ],
        "total_known_trials": total_known_trials,
        "primary_strategy_id": primary_id,
        "primary_market": primary["market"],
        "primary_timeframe": primary["timeframe"],
        "primary_economic_pass": primary["economic_pass"],
        "primary_statistical_pass": primary["statistical_pass"],
        "allowed_plateau_pbo": allowed_pbo,
        "white_reality_check_pvalue": allowed_multiple[
            "white_reality_check_pvalue"
        ],
        "hansen_spa_pvalue": allowed_multiple[
            "hansen_spa_pvalue"
        ],
        "observer": str(observer_path),
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
    }


__all__ = [
    "ORDERFLOW_DATA_BLOCKERS",
    "VOLUME_STRATEGY_ALLOWED_MARKETS",
    "VOLUME_STRATEGY_ARCHETYPES",
    "VOLUME_STRATEGY_CAMPAIGN",
    "VOLUME_STRATEGY_COORDINATES",
    "VOLUME_STRATEGY_ENGINE_VERSION",
    "VOLUME_STRATEGY_MARKETS",
    "VOLUME_STRATEGY_TIMEFRAMES",
    "VolumeBacktestBatch",
    "VolumeStrategyDNA",
    "backtest_volume_strategy_batch",
    "plan_volume_strategy_campaign",
    "run_volume_strategy_campaign",
    "volume_strategy_campaign_path",
    "volume_strategy_dna",
]
